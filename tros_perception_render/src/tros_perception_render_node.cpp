// Copyright (c) 2024，D-Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tros_perception_render_node.h"

namespace tros {

TrosPerceptionRenderNode::TrosPerceptionRenderNode(const rclcpp::NodeOptions &options) :
  Node("tros_perception_render", options) {
  RCLCPP_INFO(this->get_logger(), "TrosPerceptionRenderNode is initializing...");
  perception_topic_name_ = this->declare_parameter("perception_topic_name", perception_topic_name_);
  img_topic_name_ = this->declare_parameter("img_topic_name", img_topic_name_);
  sub_nav_grid_map_topic_name_ = this->declare_parameter("sub_nav_grid_map_topic_name", sub_nav_grid_map_topic_name_);
  sub_fusion_grid_map_topic_name_ = this->declare_parameter("sub_fusion_grid_map_topic_name", sub_fusion_grid_map_topic_name_);
  pub_render_topic_name_= this->declare_parameter("pub_render_topic_name", pub_render_topic_name_);
  pub_render_grid_map_topic_name_ = this->declare_parameter("pub_render_grid_map_topic_name", pub_render_grid_map_topic_name_);
  pub_render_perc_map_topic_name_ = this->declare_parameter("pub_render_perc_map_topic_name", pub_render_perc_map_topic_name_);
  render_sys_info_ = this->declare_parameter("render_sys_info", render_sys_info_);
  render_map_ = this->declare_parameter("render_map", render_map_);

  RCLCPP_WARN_STREAM(this->get_logger(),
    "\n             perception_topic_name [" << perception_topic_name_ << "]"
    << "\n                 img_topic_name [" << img_topic_name_ << "]"
    << "\n    sub_nav_grid_map_topic_name [" << sub_nav_grid_map_topic_name_ << "]"
    << "\n sub_fusion_grid_map_topic_name [" << sub_fusion_grid_map_topic_name_ << "]"
    << "\n          pub_render_topic_name [" << pub_render_topic_name_ << "]"
    << "\n pub_render_grid_map_topic_name [" << pub_render_grid_map_topic_name_ << "]"
    << "\n pub_render_perc_map_topic_name [" << pub_render_perc_map_topic_name_ << "]"
    << "\n                render_sys_info [" << (render_sys_info_ ? "true" : "false") << "]"
    << "\n                     render_map [" << (render_map_ ? "true" : "false") << "]"
  );

  if (render_sys_info_) {
    system_status_thread_ = std::thread([this](){
      while (rclcpp::ok()) {
        GetSystemStatus();
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    });
  }

  render_img_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
    pub_render_topic_name_,
    rclcpp::QoS(10));
  render_grid_map_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
    pub_render_grid_map_topic_name_,
    rclcpp::QoS(10));
  render_perc_map_publisher_ = this->create_publisher<sensor_msgs::msg::CompressedImage>(
    pub_render_perc_map_topic_name_,
    rclcpp::QoS(10));

  sub_perc_.subscribe(this, perception_topic_name_);
  sub_img_.subscribe(this, img_topic_name_);
  perc_synchronizer_ = std::make_shared<message_filters::Synchronizer<PercCustomSyncPolicyType>>(
    PercCustomSyncPolicyType(10), sub_perc_, sub_img_);
  perc_synchronizer_->registerCallback(std::bind(&TrosPerceptionRenderNode::PercTopicSyncCallback,
    this, std::placeholders::_1, std::placeholders::_2));

  if (render_map_) {
    sub_nav_grid_map_.subscribe(this, sub_nav_grid_map_topic_name_);
    sub_fusion_grid_map_.subscribe(this, sub_fusion_grid_map_topic_name_);
    grid_map_synchronizer_ = std::make_shared<message_filters::Synchronizer<GridMapCustomSyncPolicyType>>(
      GridMapCustomSyncPolicyType(10), sub_nav_grid_map_, sub_fusion_grid_map_);
    grid_map_synchronizer_->registerCallback(std::bind(&TrosPerceptionRenderNode::GridMapTopicSyncCallback,
      this, std::placeholders::_1, std::placeholders::_2));
    perc_map_synchronizer_ = std::make_shared<message_filters::Synchronizer<PercMapCustomSyncPolicyType>>(
      PercMapCustomSyncPolicyType(10), sub_perc_, sub_img_, sub_fusion_grid_map_);
    perc_map_synchronizer_->registerCallback(std::bind(&TrosPerceptionRenderNode::PercMapTopicSyncCallback,
      this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  }
}

void TrosPerceptionRenderNode::PercTopicSyncCallback(
  const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg_perc,
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg_img) {
  RCLCPP_INFO(this->get_logger(), "PercTopicSync Callback");
  cv::Mat mat;
  Render(msg_perc, msg_img, mat);

  std::vector<uchar> buf;  
  cv::imencode(".jpeg", mat, buf);
  auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();  
  msg->header = msg_img->header;
  msg->format = msg_img->format;  
  msg->data.assign(buf.begin(), buf.end());  
  render_img_publisher_->publish(std::move(*msg));
}  
  
void TrosPerceptionRenderNode::GridMapTopicSyncCallback(
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg_nav_grid_map,
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg_fusion_grid_map) {
  RCLCPP_INFO(this->get_logger(), "GridMapTopicSync Callback");
  if (!msg_nav_grid_map || !msg_fusion_grid_map || !render_grid_map_publisher_) {
    return;
  }

  cv::Mat mat_nav, mat_grid;
  if (RenderGridMap(msg_nav_grid_map, mat_nav, 270) == 0 &&
    RenderGridMap(msg_fusion_grid_map, mat_grid, 180) == 0) {
    // 水平拼接两个图像
    // int h = std::max(mat_nav.rows, mat_fusion.rows);
    // int w = mat_nav.cols + mat_fusion.cols;
    // // 创建一个源矩阵  
    // cv::Mat src = cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    // RCLCPP_DEBUG(this->get_logger(), "RenderGridMap: w=%d, h=%d", w, h);
    // // 将源矩阵复制到目标矩阵  
    // mat_nav.copyTo(src(cv::Rect(0, 0, mat_nav.cols, mat_nav.rows)));  
    // mat_fusion.copyTo(src(cv::Rect(mat_nav.cols, 0,
    //   mat_fusion.cols, mat_fusion.rows)));

    // 对mat做 resize
    {
      float ratio = 640.0 / static_cast<float>(mat_nav.cols);
      int newWidth = static_cast<float>(mat_nav.cols) * ratio;  
      int newHeight = static_cast<float>(mat_nav.rows) * ratio;  
      cv::resize(mat_nav, mat_nav, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    }
    {
      float ratio = 640.0 / static_cast<float>(mat_grid.cols);
      int newWidth = static_cast<float>(mat_grid.cols) * ratio;  
      int newHeight = static_cast<float>(mat_grid.rows) * ratio;  
      cv::resize(mat_grid, mat_grid, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    }
    
    // 上下拼接
    int h = mat_nav.rows + mat_grid.rows;
    int w = std::max(mat_nav.cols, mat_grid.cols);
    // 拼接后的mat
    cv::Mat mat_fusion;
    mat_fusion = cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
    // 将源矩阵复制到目标矩阵  
    mat_nav.copyTo(mat_fusion(cv::Rect(0, 0, mat_nav.cols, mat_nav.rows)));  
    mat_grid.copyTo(mat_fusion(cv::Rect(0, mat_nav.rows, mat_grid.cols, mat_grid.rows)));

    // 渲染时间戳
    std::string timestamp_str = std::to_string(msg_nav_grid_map->header.stamp.sec) + std::string(".") +
      std::to_string(msg_nav_grid_map->header.stamp.nanosec);
    cv::putText(mat_fusion,
                timestamp_str,
                cv::Point2f(10, 30),
                cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2.0);  

    std::vector<uchar> buf;  
    cv::imencode(".jpeg", mat_fusion, buf);
    auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();  
    msg->header = msg_nav_grid_map->header;
    msg->format = "jpeg";  
    msg->data.assign(buf.begin(), buf.end());  
    render_grid_map_publisher_->publish(std::move(*msg));
  }
}  

void TrosPerceptionRenderNode::PercMapTopicSyncCallback(
    ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg_perc,
    sensor_msgs::msg::CompressedImage::ConstSharedPtr msg_img,
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg_fusion_grid_map) {
  RCLCPP_INFO(this->get_logger(), "PercMapTopicSync Callback");
  if (!msg_perc || !msg_img || !msg_fusion_grid_map || !render_perc_map_publisher_) {
    return;
  }

  // 感知和图像渲染
  cv::Mat mat_perc;
  Render(msg_perc, msg_img, mat_perc);

  // map转成的mat
  cv::Mat mat_map;
  if (RenderGridMap(msg_fusion_grid_map, mat_map, 180) != 0) {
    return;
  }
  // 对mat做 resize
  float ratio = static_cast<float>(mat_perc.cols) / static_cast<float>(mat_map.cols);
  int newWidth = static_cast<float>(mat_map.cols) * ratio;  
  int newHeight = static_cast<float>(mat_map.rows) * ratio;  
  cv::resize(mat_map, mat_map, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  // 渲染网格，原始grid分辨率为 grid->info.resolution
  // 10个像素偏移渲染一条线
  int num_pixels = 10;
  float map_size = static_cast<float>(num_pixels) * msg_fusion_grid_map->info.resolution;
  num_pixels = static_cast<float>(num_pixels) * ratio;
  for (int i=num_pixels; i<mat_map.rows; i+=num_pixels) {
    // 画横线
    cv::line(mat_map, cv::Point(0, i), cv::Point(mat_map.cols, i), cv::Scalar(255, 0, 0), 1);
  }
  for (int j=num_pixels; j<mat_map.cols; j+=num_pixels) {
    // 画竖线
    cv::line(mat_map, cv::Point(j, 0), cv::Point(j, mat_map.rows), cv::Scalar(255, 0, 0), 1);
  }
  // 渲染size 信息
  std::string map_size_info = tools_.FloatToString(map_size) + " meters";
  cv::putText(mat_map,
              map_size_info,
              cv::Point2f(10, 70),
              cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
              1.0,
              cv::Scalar(255, 0, 0),
              2.0);  

  // 拼接后的mat
  cv::Mat mat_fusion;
  // 上下拼接
  int h = mat_perc.rows + mat_map.rows;
  int w = std::max(mat_perc.cols, mat_map.cols);
  mat_fusion = cv::Mat(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
  // 将源矩阵复制到目标矩阵  
  mat_perc.copyTo(mat_fusion(cv::Rect(0, 0, mat_perc.cols, mat_perc.rows)));  
  mat_map.copyTo(mat_fusion(cv::Rect(0, mat_perc.rows, mat_map.cols, mat_map.rows)));

  std::vector<uchar> buf;  
  cv::imencode(".jpeg", mat_fusion, buf);
  auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();  
  msg->header = msg_perc->header;
  msg->format = "jpeg";  
  msg->data.assign(buf.begin(), buf.end());
  RCLCPP_INFO(this->get_logger(), "Publish perc map with topic: %s", pub_render_perc_map_topic_name_.data());
  render_perc_map_publisher_->publish(std::move(*msg));
}  


cv::Mat TrosPerceptionRenderNode::compressedImageToMat(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr img_ptr) {  
    try {  
        // 确保图像数据不是空的  
        if (!img_ptr || img_ptr->data.empty()) {  
            std::cerr << "Compressed image is empty!" << std::endl;  
            return cv::Mat();  
        }  
  
        // 根据压缩格式创建解码器  
        std::vector<uchar> buf(img_ptr->data.begin(), img_ptr->data.end());  
        cv::Mat img_decoded;  
  
        // 假设数据是JPEG格式的，这是最常见的格式  
        if (img_ptr->format == "jpeg") {  
            cv::imdecode(buf, cv::IMREAD_COLOR, &img_decoded);  
        } else if (img_ptr->format == "png") {  
            cv::imdecode(buf, cv::IMREAD_UNCHANGED, &img_decoded);  
        } else {  
            std::cerr << "Unsupported image format: " << img_ptr->format << std::endl;  
            return cv::Mat();  
        }  
  
        // 如果解码失败，返回空的Mat  
        if (img_decoded.empty()) {  
            std::cerr << "Failed to decode image!" << std::endl;  
            return cv::Mat();  
        }  
  
        return img_decoded;  
    } catch (const std::exception& e) {  
        std::cerr << "Error processing image: " << e.what() << std::endl;  
        return cv::Mat();  
    }  
}

int TrosPerceptionRenderNode::Render(
  const ai_msgs::msg::PerceptionTargets::ConstSharedPtr msg_perc,
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg_img,
  cv::Mat &mat) {
  if (!msg_perc || !msg_img) return -1;
  mat = compressedImageToMat(msg_img);  
  if (mat.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to decode image.");
    return -1;
  }

  RCLCPP_INFO(this->get_logger(),
              "target size: %ld",
              msg_perc->targets.size());
  for (size_t idx = 0; idx < msg_perc->targets.size(); idx++) {
    const auto &target = msg_perc->targets.at(idx);
    RCLCPP_INFO(this->get_logger(),
                "target type: %s, track_id: %ld, rois.size: %ld",
                target.type.c_str(),
                target.track_id,
                target.rois.size());
    auto &color = colors[idx % colors.size()];
    for (const auto &roi : target.rois) {
      RCLCPP_INFO(
          this->get_logger(),
          "roi.type: %s, x_offset: %d y_offset: %d width: %d height: %d",
          roi.type.c_str(),
          roi.rect.x_offset,
          roi.rect.y_offset,
          roi.rect.width,
          roi.rect.height);
      cv::rectangle(mat,
                    cv::Point(roi.rect.x_offset, roi.rect.y_offset),
                    cv::Point(roi.rect.x_offset + roi.rect.width,
                              roi.rect.y_offset + roi.rect.height),
                    color,
                    3);
      std::string roi_type = target.type;
      if (!roi.type.empty()) {
        roi_type = roi.type;
      }
      if (!roi_type.empty()) {
        cv::putText(mat,
                    roi_type + "_" + std::to_string(target.track_id),
                    cv::Point2f(roi.rect.x_offset, roi.rect.y_offset - 10),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1.5);
      }
    }

    if (!target.rois.empty()) {
      int pt_x = target.rois.front().rect.x_offset;
      int pt_y = target.rois.front().rect.y_offset;
      int y_offset = 0;
      auto &color_attr = colors.at(std::min(2, static_cast<int>(colors.size() - 1)));
      // 过滤需要可视化的属性
      PoseAttribute pose_attr;
      for (const auto& attr : target.attributes) {
        if (attr.type == "width_cm") {
          pose_attr.width = attr.value / 100.0;
        } else if (attr.type == "height_cm") {
          pose_attr.height = attr.value / 100.0;
        } else if (attr.type == "x_cm") {
          pose_attr.x = attr.value / 100.0;
        } else if (attr.type == "y_cm") {
          pose_attr.y = attr.value / 100.0;
        } else if (attr.type == "z_cm") {
          pose_attr.z = attr.value / 100.0;
        }
      }
      // Score is the person ROI's confidence, not an attribute.
      for (const auto& roi : target.rois) {
        if (roi.type == "person") {
          pose_attr.score = roi.confidence;
          break;
        }
      }

      if (!std::isnan(pose_attr.x) && !std::isnan(pose_attr.y) && !std::isnan(pose_attr.z)) {
        y_offset += 30;
        // int render_y = pt_y + y_offset;
        // if (render_y < 0 ) render_y = 0;
        // if (render_y > mat.rows) render_y = mat.rows;
        cv::putText(mat,
                    "xy_cam (" + tools_.FloatToString(pose_attr.x) + ", " + tools_.FloatToString(pose_attr.y) + ")",
                    cv::Point2f(pt_x, pt_y + y_offset),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color_attr,
                    2.0);
      }
      if (!std::isnan(pose_attr.width) || !std::isnan(pose_attr.height)) {
        y_offset += 30;
        std::string wh = "wh (";
        wh += std::isnan(pose_attr.width) ? "nan" : tools_.FloatToString(pose_attr.width);
        wh += ", ";
        wh += std::isnan(pose_attr.height) ? "nan" : tools_.FloatToString(pose_attr.height);
        wh += ")";
        cv::putText(mat,
                    wh,
                    cv::Point2f(pt_x, pt_y + y_offset),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color_attr,
                    2.0);
      }
      {
        y_offset += 30;
        cv::putText(mat,
                    "id (" + std::to_string(target.track_id) + ")",
                    cv::Point2f(pt_x, pt_y + y_offset),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color_attr,
                    2.0);
      }
      if (!std::isnan(pose_attr.score)) {
        y_offset += 30;
        cv::putText(mat,
                    "score (" + tools_.FloatToString(pose_attr.score) + ")",
                    cv::Point2f(pt_x, pt_y + y_offset),
                    cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color_attr,
                    2.0);
      }
    }

    for (const auto &lmk : target.points) {
      for (const auto &pt : lmk.point) {
        cv::circle(mat, cv::Point(pt.x, pt.y), 3, color, 3);
      }
    }

    if (!target.captures.empty()) {
      float alpha_f = 0.5;
      for (auto capture: target.captures) {
        int parsing_width = capture.img.width;
        int parsing_height = capture.img.height;
        cv::Mat parsing_img(parsing_height, parsing_width, CV_8UC3);
        uint8_t *parsing_img_ptr = parsing_img.ptr<uint8_t>();

        for (int h = 0; h < parsing_height; ++h) {
          for (int w = 0; w < parsing_width; ++w) {
            auto id = static_cast<size_t>(capture.features[h * parsing_width + w]);
            id = id >= 80? (id % 80) + 1: id;
            *parsing_img_ptr++ = bgr_putpalette[id * 3];
            *parsing_img_ptr++ = bgr_putpalette[id * 3 + 1];
            *parsing_img_ptr++ = bgr_putpalette[id * 3 + 2];
          }
        }

        cv::resize(parsing_img, parsing_img, mat.size(), 0, 0);
        // alpha blending
        cv::Mat dst;
        addWeighted(mat, alpha_f, parsing_img, 1 - alpha_f, 0.0, dst);
        mat = std::move(dst);
      }
    }
  }

  if (render_sys_info_) {
    // 渲染时间戳
    std::string timestamp_str = std::to_string(msg_img->header.stamp.sec) + std::string(".") +
      std::to_string(msg_img->header.stamp.nanosec);
    cv::putText(mat,
                timestamp_str,
                cv::Point2f(10, 30),
                cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2.0);  

    // 渲染系统信息
    std::string system_status =
      "CPU: " + tools_.FloatToString(system_status_->cpu_usage) + "%" +
      ", Temp: " + system_status_->temperature;
    cv::putText(mat,
                system_status,
                cv::Point2f(10, 70),
                cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2.0);  
  }

  // std::string saving_path = "render_" + msg_perc->header.frame_id + "_" +
  //                           std::to_string(msg_perc->header.stamp.sec) + "_" +
  //                           std::to_string(msg_perc->header.stamp.nanosec) +
  //                           ".jpeg";
  // RCLCPP_WARN(this->get_logger(),
  //             "Draw result to file: %s",
  //             saving_path.c_str());
  // cv::imwrite(saving_path, mat);
  return 0;
}
  
void TrosPerceptionRenderNode::GetSystemStatus() {
  float cpu_usage = tools_.GetCPUUsage(8);
  std::string cpu_temperature = tools_.GetCPUTemperature();
  RCLCPP_DEBUG_STREAM(this->get_logger(),
  "CPU Usage: " << cpu_usage << "%"
  ", temperature: " << cpu_temperature);  
  
  system_status_->cpu_usage = cpu_usage;
  system_status_->temperature = cpu_temperature;
}

int TrosPerceptionRenderNode::RenderGridMap(
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr grid, cv::Mat& mat, int rotate_degree) {
  if (!grid) {
    return -1;
  }
  /*
  grid的坐标：
                  X  cosatmap index
                  ^  ^
                  |  |W-1
                  |  .
                  |W .
                  |  .
                  |  1
  Y<-------------  0
          H
  */
  // 创建一个 grid->info.height X grid->info.width 尺寸的cv::Mat
  // 60 X 40
  mat = cv::Mat(grid->info.width, grid->info.height, CV_8UC3, cv::Scalar(255, 255, 255));
  RCLCPP_DEBUG(this->get_logger(),
    "grid w: %d, h: %d, mat cols: %d, rows: %d", grid->info.width, grid->info.height, mat.cols, mat.rows);

  for (int i=0; i<mat.rows; i++) {
    for (int j=0; j<mat.cols; j++) {
      int idx_grid = j * grid->info.width + i;
      int cost = grid->data[idx_grid];

      cv::Vec3b color(128, 128, 128);
      // static constexpr int8_t OCC_GRID_UNKNOWN = -1;
      // static constexpr int8_t OCC_GRID_FREE = 0;
      // static constexpr int8_t OCC_GRID_OCCUPIED = 100;
      if (cost == 100) {
        // OCC_GRID_OCCUPIED
        color[0] = 0; // B  
        color[1] = 0; // G  
        color[2] = 255; // R  
      } else if (cost == -1) {
        // OCC_GRID_UNKNOWN
      } else {
        // OCC_GRID_FREE
        color[0] = 255; // B  
        color[1] = 255; // G  
        color[2] = 255; // R  
      }

      mat.at<cv::Vec3b>(i, j) = color;
    }
  }

  // // 渲染网格，原始grid分辨率为 grid->info.resolution
  // // 10个像素偏移渲染一条线
  // for (int i=10; i<mat.rows; i+=10) {
  //   // 画横线
  //   cv::line(mat, cv::Point(0, i), cv::Point(mat.cols, i), cv::Scalar(255, 0, 0), 1);
  // }
  // for (int j=10; j<mat.cols; j+=10) {
  //   // 画竖线
  //   cv::line(mat, cv::Point(j, 0), cv::Point(j, mat.rows), cv::Scalar(255, 0, 0), 1);
  // }

  // // 对mat做10倍resize
  // int newWidth = mat.cols * 10;  
  // int newHeight = mat.rows * 10;  
  // cv::resize(mat, mat, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
  
  // if (180 == rotate_degree) {
  //   // 使用flip函数进行180度旋转  
  //   // 首先沿X轴翻转（水平翻转），然后沿Y轴翻转（垂直翻转）  
  //   // 或者直接使用flipCode = -1（同时沿X轴和Y轴翻转）  
  //   cv::flip(mat, mat, -1);  
  // } else if (90 == rotate_degree) {
  //   // 顺时针旋转90度  
  //   cv::Mat img_rotated;  
  //   cv::transpose(mat, img_rotated); // 转置  
  //   cv::flip(img_rotated, mat, 1); // 绕y轴翻转 
  // } else if (-90 == rotate_degree) {
  //   // 逆时针旋转90度
  //   cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
  // }
  
  // cv::ROTATE_90_CLOCKWISE：逆时针旋转90度。
  // cv::ROTATE_180：旋转180度。
  // cv::ROTATE_90_COUNTERCLOCKWISE：顺时针旋转90度。
  if (90 == rotate_degree) {
    cv::rotate(mat, mat, cv::ROTATE_90_COUNTERCLOCKWISE);
  } else if (180 == rotate_degree) {
    cv::rotate(mat, mat, cv::ROTATE_180);
  } else if (270 == rotate_degree) {
    cv::rotate(mat, mat, cv::ROTATE_90_CLOCKWISE);
  }
  // // 渲染时间戳
  // std::string timestamp_str = std::to_string(grid->header.stamp.sec) + std::string(".") +
  //   std::to_string(grid->header.stamp.nanosec);
  // cv::putText(mat,
  //             timestamp_str,
  //             cv::Point2f(10, 30),
  //             cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
  //             1.0,
  //             cv::Scalar(0, 255, 0),
  //             2.0);  
              
  // static bool dump = true;
  // if (dump) 
  // {
  //   dump = false;
  //   std::string saving_path = "render_fusiongrid_" + std::to_string(mat.cols) +
  //     "_" + std::to_string(mat.rows) + "_"
  //     + std::to_string(grid->header.stamp.sec) + "_"
  //     + std::to_string(grid->header.stamp.nanosec)
  //     + ".jpeg";
  //   RCLCPP_WARN(this->get_logger(),
  //               "Draw result to file: %s",
  //               saving_path.c_str());
  //   cv::imwrite(saving_path, mat);
  // }
  
  return 0;
}

} // namespace

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(tros::TrosPerceptionRenderNode)