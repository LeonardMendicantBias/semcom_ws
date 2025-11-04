#include <chrono>
#include <functional>
#include <memory>
#include <fstream>
#include <string>
#include <cstdlib>
#include <deque>

#include "rclcpp/rclcpp.hpp"

#include <cv_bridge/cv_bridge.h>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "ament_index_cpp/get_package_share_directory.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "semcom_msgs/msg/code.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;
using namespace nvinfer1;


// Helper: check CUDA errors
#define CUDA_CHECK(call) do { cudaError_t e = call; if(e != cudaSuccess){ \
  RCLCPP_FATAL(rclcpp::get_logger("tensorrt_node"), "CUDA err %s", cudaGetErrorString(e)); 

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

struct BindingInfo {
  int index;
  bool isInput;
  nvinfer1::Dims dims; // static dims; for dynamic shapes you'll call context->setBindingDimensions
  size_t sizeBytes;
  void* devicePtr = nullptr;
  std::string name;
};

// Utility to load engine file
std::vector<char> loadEngineFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open engine file");
    size_t size = file.tellg();
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

std::string dataTypeToString(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return "float";
        case nvinfer1::DataType::kHALF: return "half";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
        // Add other data types if they exist in your TensorRT version
        default: return "a";
    }
}

class SemPublisher : public rclcpp::Node
{
	public:
		SemPublisher() : Node("Semantic_Node")
		{
			subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
				"/camera/camera/color/image_raw", 1,
				std::bind(&SemPublisher::imageCallback, this, _1)
			);

			publisher_ = this->create_publisher<semcom_msgs::msg::Code>(
				"/camera/camera/color/image_code", 1
			);
			
			// Load TensorRT engine
			std::string package_share_dir = ament_index_cpp::get_package_share_directory("cpp_commun");
			std::string engine_path = package_share_dir + "/models/encoder.trt";
			auto engine_data = loadEngineFile(engine_path);

			Logger logger;
			runtime_ = nvinfer1::createInferRuntime(logger);
			engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
			context_ = engine_->createExecutionContext();
			
			int nbBindings = engine_->getNbIOTensors();
			buffers_.resize(nbBindings);
			binding_sizes_.resize(nbBindings);

			for (int i = 0; i < nbBindings; ++i) {
				auto type_ = engine_->getBindingDataType(i);
    			const char* name = engine_->getBindingName(i);
				nvinfer1::Dims dims = context_->getBindingDimensions(i);
				size_t vol = 1;

				// Determine if this binding has a dynamic sequence axis
				bool is_dynamic_sequence = (i == 1 || i == 2 || i == 3 || i == 4);

				for (int j = 0; j < dims.nbDims; ++j) {
					int dim = dims.d[j];
					if (is_dynamic_sequence && j == 3) {
						dim = max_sequence_;  // use max sequence length
					}
					RCLCPP_INFO(this->get_logger(), "dim: %zu", dim);
					vol *= dim;
				}
				
				size_t elem_size = 4; // default float/int32
				if (i == 6) elem_size = 2; // mask is boolean
				if (i == 5) elem_size = 2; // code is int16

				binding_sizes_[i] = vol * elem_size;
				RCLCPP_INFO(this->get_logger(), "size of %s: %zu in %s", name, vol*elem_size, dataTypeToString(type_).c_str());
				// cudaMalloc(&buffers_[i], binding_sizes_[i]);
				cudaError_t err = cudaMalloc(&buffers_[i], binding_sizes_[i]);
				if (err != cudaSuccess) {
					RCLCPP_ERROR(this->get_logger(), "cudaMalloc failed for binding %d", i);
					throw std::runtime_error("cudaMalloc failed");
				}
			}
			
			// Allocate CPU sliding window buffers
			prev_keys_window_.resize(max_sequence_);
			prev_values_window_.resize(max_sequence_);
			for (int i = 0; i < max_sequence_; ++i) {
				// RCLCPP_INFO(this->get_logger(), "a: %zu, b: %zu", binding_sizes_[5], binding_sizes_[6]);
				prev_keys_window_[i].resize(binding_sizes_[3]/sizeof(float)/max_sequence_, 0.0f);
				prev_values_window_[i].resize(binding_sizes_[4]/sizeof(float)/max_sequence_, 0.0f);
			}
			
			prev_keys_gpu_.resize(binding_sizes_[1]/4, 0.0f);
			prev_values_gpu_.resize(binding_sizes_[2]/4, 0.0f);
		}

		~SemPublisher() {
			for (void* buf : buffers_) cudaFree(buf);
			context_->destroy();
			engine_->destroy();
			runtime_->destroy();
		}

	private:
  		void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
			// Convert ROS image to float32 normalized [0,1]
			cv::Mat image = cv_bridge::toCvShare(msg, "rgb8")->image;

			nvinfer1::Dims frames_dims = context_->getBindingDimensions(0);
			int H = frames_dims.d[3];
			int W = frames_dims.d[4];
			// RCLCPP_INFO(this->get_logger(), "H: %zu, W: %zu", H, W);
			
			cv::Mat image_resized;
			cv::resize(image, image_resized, cv::Size(W, H));
			cv::Mat image_float;
			image_resized.convertTo(image_float, CV_32FC3, 1.0 / 255.0);

			std::vector<float> img_chw(image_float.total() * 3);
			for (int c = 0; c < 3; ++c)
				for (int i = 0; i < H; ++i)
					for (int j = 0; j < W; ++j)
						img_chw[c * H * W + i * W + j] = image_float.at<cv::Vec3f>(i,j)[c];
			
			if (sequence_length_ < max_sequence_) {
				prev_keys_window_.pop_front();
				prev_values_window_.pop_front();
			} else {
				sequence_length_++;
			}
			// Push placeholder for current frame (will be updated after inference)
			prev_keys_window_.emplace_back(std::vector<float>(prev_keys_window_[0].size(), 0.0f));
			prev_values_window_.emplace_back(std::vector<float>(prev_values_window_[0].size(), 0.0f));
			
			prepareSlidingWindowBuffers();

			// Copy image to GPU
			cudaMemcpy(buffers_[0], img_chw.data(), binding_sizes_[0], cudaMemcpyHostToDevice);
			// Copy previous keys/values to GPU
			cudaMemcpy(buffers_[1], prev_keys_gpu_.data(), binding_sizes_[1], cudaMemcpyHostToDevice);
        	cudaMemcpy(buffers_[2], prev_values_gpu_.data(), binding_sizes_[2], cudaMemcpyHostToDevice);

			// Run inference
			context_->executeV2(buffers_.data());

			std::vector<float> cur_keys(binding_sizes_[3]/sizeof(float));
			std::vector<float> cur_values(binding_sizes_[4]/sizeof(float));
			std::vector<int32_t> code(binding_sizes_[5]/sizeof(int32_t));
			std::vector<int32_t> mask(binding_sizes_[6]/sizeof(int32_t));

			cudaMemcpy(cur_keys.data(), buffers_[3], binding_sizes_[3], cudaMemcpyDeviceToHost);
			cudaMemcpy(cur_values.data(), buffers_[4], binding_sizes_[4], cudaMemcpyDeviceToHost);
			cudaMemcpy(code.data(), buffers_[5], binding_sizes_[5], cudaMemcpyDeviceToHost);
			cudaMemcpy(mask.data(), buffers_[6], binding_sizes_[6], cudaMemcpyDeviceToHost);

			// Save current keys/values for next inference
			// prev_keys_ = cur_keys;
			// prev_values_ = cur_values;
        	updateSlidingWindow(cur_keys, cur_values);
			// RCLCPP_INFO(this->get_logger(), "Sliding windows updated");

			// Example: log code and mask sizes
			RCLCPP_INFO(this->get_logger(), "Inference done: code size %zu, mask size %zu, ks size %zu, vs size %zu", code.size(), mask.size(), cur_keys.size(), cur_values.size());
			publishResult(mask, code);
		}

		void publishResult(
			const std::vector<int32_t>& mask,
			const std::vector<int32_t>& code)
		{
			semcom_msgs::msg::Code message_;
			message_.header.stamp = rclcpp::Clock().now();

			// Find indices where mask is true
			std::vector<int32_t> masked_codes;
			for (size_t i = 0; i < mask.size(); ++i) {
				if (mask[i]) masked_codes.push_back(code[i]);
			}
			message_.length = static_cast<uint16_t>(masked_codes.size());

			// Calculate total bits and bytes
			size_t total_bits = masked_codes.size() * 13;
			size_t total_bytes = (total_bits + 7) / 8;
			message_.data.resize(total_bytes+1024, 0);

			size_t bit_pos = 0;
			for (size_t i = 0; i < masked_codes.size(); ++i) {
				uint16_t val = static_cast<uint16_t>(masked_codes[i] & 0x1FFF); // 13-bit mask
				for (int b = 0; b < 13; ++b, ++bit_pos) {
					if ((val >> b) & 1)
						message_.data[bit_pos / 8] |= (1 << (bit_pos % 8));
				}
			}

			publisher_->publish(message_);
			// RCLCPP_INFO(this->get_logger(), "Publishing");
		}
		
		void prepareSlidingWindowBuffers() {
			// Flatten deque into contiguous GPU buffer
			size_t offset = 0;
			for (int i=0; i<sequence_length_; ++i) {
				std::memcpy(prev_keys_gpu_.data()+offset, prev_keys_window_[i].data(),
							prev_keys_window_[i].size()*sizeof(float));
				std::memcpy(prev_values_gpu_.data()+offset, prev_values_window_[i].data(),
							prev_values_window_[i].size()*sizeof(float));
				offset += prev_keys_window_[i].size();
			}
			// Zero-pad remaining
			size_t remaining = binding_sizes_[1]/sizeof(float) - offset;
			if (remaining>0) std::memset(prev_keys_gpu_.data()+offset, 0, remaining*sizeof(float));
			remaining = binding_sizes_[2]/sizeof(float) - offset;
			if (remaining>0) std::memset(prev_values_gpu_.data()+offset, 0, remaining*sizeof(float));
		}

		void updateSlidingWindow(const std::vector<float>& cur_k, const std::vector<float>& cur_v) {
			prev_keys_window_.back() = cur_k;
			prev_values_window_.back() = cur_v;
		}

		rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
		rclcpp::Publisher<semcom_msgs::msg::Code>::SharedPtr publisher_;

		std::vector<BindingInfo> bindingInfos_;
		std::unordered_map<std::string,int> nameToIndex_;
  		cudaStream_t stream_;

		std::vector<void*> buffers_;
		std::vector<size_t> binding_sizes_;
		
		std::deque<std::vector<float>> prev_keys_window_;
		std::deque<std::vector<float>> prev_values_window_;
		std::vector<float> prev_keys_gpu_;
		std::vector<float> prev_values_gpu_;

		nvinfer1::IRuntime* runtime_;
		nvinfer1::ICudaEngine* engine_;
		nvinfer1::IExecutionContext* context_;
    	
		int sequence_length_ = 0;
    	int max_sequence_ = 16;
};

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<SemPublisher>());
	rclcpp::shutdown();
	return 0;
}
