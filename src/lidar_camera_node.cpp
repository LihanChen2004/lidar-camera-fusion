#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_spherical.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include <armadillo>
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/impl/point_types.hpp>

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

// Publisher
ros::Publisher pcOnimg_pub;
ros::Publisher pc_pub;

float maxlen = 100.0;  // maxima distancia del lidar
float minlen = 0.01;   // minima distancia del lidar
float max_FOV = 3.0;   // en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;   // en radianes angulo minimo de vista de la camara

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x = 0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width = 360.0f;
float max_angle_height = 180.0f;
float z_max = 100.0f;
float z_min = 100.0f;

float max_depth = 100.0;
float min_depth = 8.0;

float interpol_value = 20.0;

// input topics
std::string imgTopic = "/camera/color/image_raw";
std::string pcTopic = "/velodyne_points";

// matrix calibration lidar and camera
Eigen::MatrixXf Tlc(3, 1);  // translation matrix lidar-camera
Eigen::MatrixXf Rlc(3, 3);  // rotation matrix lidar-camera
Eigen::MatrixXf Mc(3, 4);   // camera calibration matrix

// range image parametros
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

///////////////////////////////////////callback

void callback(
  const boost::shared_ptr<const sensor_msgs::PointCloud2> & input_cloud,
  const ImageConstPtr & input_image)
{
  // 图像数据转换
  cv_bridge::CvImagePtr cv_image_ptr, color_image_ptr;
  try {
    cv_image_ptr = cv_bridge::toCvCopy(input_image, sensor_msgs::image_encodings::BGR8);
    color_image_ptr = cv_bridge::toCvCopy(input_image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception & e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // 点云数据转换 sensor_msgs::PointCloud2 -> pcl::PointCloud<pcl::PointXYZI>
  pcl::PCLPointCloud2 pcl_pc2;
  pcl_conversions::toPCL(*input_cloud, pcl_pc2);
  PointCloud::Ptr original_cloud(new PointCloud);
  pcl::fromPCLPointCloud2(pcl_pc2, *original_cloud);

  // 点云过滤，移除点云中的 NaN 点，并根据距离过滤点云。
  if (original_cloud == nullptr) return;

  PointCloud::Ptr filtered_cloud(new PointCloud);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*original_cloud, *filtered_cloud, indices);

  PointCloud::Ptr distance_filtered_cloud(new PointCloud);
  for (const auto & point : filtered_cloud->points) {
    double distance = std::sqrt(point.x * point.x + point.y * point.y);
    if (distance < minlen || distance > maxlen) continue;
    distance_filtered_cloud->push_back(point);
  }

  // 点云投影到图像
  Eigen::Affine3f sensor_pose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
  rangeImage->pcl::RangeImage::createFromPointCloud(
    *distance_filtered_cloud, pcl::deg2rad(angular_resolution_x),
    pcl::deg2rad(angular_resolution_y), pcl::deg2rad(max_angle_width),
    pcl::deg2rad(max_angle_height), sensor_pose, coordinate_frame, 0.0f, 0.0f, 0);

  int img_width = rangeImage->width;
  int img_height = rangeImage->height;

  arma::mat range_matrix(img_height, img_width, arma::fill::zeros);
  arma::mat height_matrix(img_height, img_width, arma::fill::zeros);

  for (int i = 0; i < img_width; ++i) {
    for (int j = 0; j < img_height; ++j) {
      float range = rangeImage->getPoint(i, j).range;
      float height = rangeImage->getPoint(i, j).z;

      if (std::isinf(range) || range < minlen || range > maxlen || std::isnan(height)) {
        continue;
      }
      range_matrix(j, i) = range;
      height_matrix(j, i) = height;
    }
  }

  // 图像插值
  arma::vec X = arma::regspace(1, range_matrix.n_cols);
  arma::vec Y = arma::regspace(1, range_matrix.n_rows);
  arma::vec XI = arma::regspace(X.min(), 1.0, X.max());
  arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max());
  arma::mat interpolated_range_matrix;
  arma::mat interpolated_height_matrix;

  arma::interp2(X, Y, range_matrix, XI, YI, interpolated_range_matrix, "lineal");
  arma::interp2(X, Y, height_matrix, XI, YI, interpolated_height_matrix, "lineal");

  // 过滤与背景插值的元素
  arma::mat filtered_interpolated_range_matrix = interpolated_range_matrix;

  for (uint i = 0; i < interpolated_range_matrix.n_rows; i++) {
    for (uint j = 0; j < interpolated_range_matrix.n_cols; j++) {
      if (interpolated_range_matrix(i, j) == 0) {
        if (i + interpol_value < interpolated_range_matrix.n_rows) {
          for (int k = 1; k <= interpol_value; k++) {
            filtered_interpolated_range_matrix(i + k, j) = 0;
          }
        }
        if (i > interpol_value) {
          for (int k = 1; k <= interpol_value; k++) {
            filtered_interpolated_range_matrix(i - k, j) = 0;
          }
        }
      }
    }
  }

  // 将范围图像转换为点云
  PointCloud::Ptr output_cloud(new PointCloud);
  output_cloud->width = interpolated_range_matrix.n_cols;
  output_cloud->height = interpolated_range_matrix.n_rows;
  output_cloud->is_dense = false;
  output_cloud->points.resize(output_cloud->width * output_cloud->height);

  int num_points = 0;
  for (uint i = 0; i < interpolated_range_matrix.n_rows - interpol_value; i++) {
    for (uint j = 0; j < interpolated_range_matrix.n_cols; j++) {
      float angle = M_PI - ((2.0 * M_PI * j) / (interpolated_range_matrix.n_cols));

      if (angle < min_FOV - M_PI / 2.0 || angle > max_FOV - M_PI / 2.0) continue;

      if (filtered_interpolated_range_matrix(i, j) != 0) {
        float distance = filtered_interpolated_range_matrix(i, j);
        float x = std::sqrt(std::pow(distance, 2) - std::pow(interpolated_height_matrix(i, j), 2)) *
                  std::cos(angle);
        float y = std::sqrt(std::pow(distance, 2) - std::pow(interpolated_height_matrix(i, j), 2)) *
                  std::sin(angle);

        float lidar_angle_correction = 0.6 * M_PI / 180.0;

        Eigen::MatrixXf lidar_rotation_matrix(3, 3);
        lidar_rotation_matrix << std::cos(lidar_angle_correction), 0,
          std::sin(lidar_angle_correction), 0, 1, 0, -std::sin(lidar_angle_correction), 0,
          std::cos(lidar_angle_correction);

        Eigen::Vector3f point_in_lidar_frame(x, y, interpolated_height_matrix(i, j));
        point_in_lidar_frame = lidar_rotation_matrix * point_in_lidar_frame;

        output_cloud->points[num_points].x = point_in_lidar_frame.x();
        output_cloud->points[num_points].y = point_in_lidar_frame.y();
        output_cloud->points[num_points].z = point_in_lidar_frame.z();

        num_points++;
      }
    }
  }

  // clang-format off
  Eigen::Matrix4f lidar_to_camera_transform;
  lidar_to_camera_transform << Rlc(0), Rlc(3), Rlc(6), Tlc(0),
                               Rlc(1), Rlc(4), Rlc(7), Tlc(1),
                               Rlc(2), Rlc(5), Rlc(8), Tlc(2),
                               0, 0, 0, 1;
  // clang-format on

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  for (int i = 0; i < num_points; i++) {
    Eigen::Vector4f point_in_lidar_frame(
      -output_cloud->points[i].y, -output_cloud->points[i].z, output_cloud->points[i].x, 1.0);

    Eigen::Vector3f point_in_camera_frame = Mc * (lidar_to_camera_transform * point_in_lidar_frame);

    int image_x = static_cast<int>(point_in_camera_frame.x() / point_in_camera_frame.z());
    int image_y = static_cast<int>(point_in_camera_frame.y() / point_in_camera_frame.z());

    if (
      image_x < 0 || image_x >= input_image->width || image_y < 0 ||
      image_y >= input_image->height) {
      continue;
    }

    auto & color = color_image_ptr->image.at<cv::Vec3b>(image_y, image_x);

    pcl::PointXYZRGB colored_point;
    colored_point.x = output_cloud->points[i].x;
    colored_point.y = output_cloud->points[i].y;
    colored_point.z = output_cloud->points[i].z;
    colored_point.r = color[2];
    colored_point.g = color[1];
    colored_point.b = color[0];

    colored_point_cloud->points.push_back(colored_point);

    int color_intensity_x = static_cast<int>(255 * (output_cloud->points[i].x / maxlen));
    int color_intensity_z =
      std::min(static_cast<int>(255 * (output_cloud->points[i].x / 10.0)), 255);

    cv::circle(
      cv_image_ptr->image, cv::Point(image_x, image_y), 1,
      CV_RGB(255 - color_intensity_x, color_intensity_z, color_intensity_x), cv::FILLED);
  }

  colored_point_cloud->is_dense = true;
  colored_point_cloud->width = static_cast<int>(colored_point_cloud->points.size());
  colored_point_cloud->height = 1;
  colored_point_cloud->header.frame_id = original_cloud->header.frame_id;

  pcOnimg_pub.publish(cv_image_ptr->toImageMsg());
  pc_pub.publish(colored_point_cloud);
}

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "pontCloudOntImage");
  ros::NodeHandle nh;

  /// Load Parameters
  nh.getParam("/maxlen", maxlen);
  nh.getParam("/minlen", minlen);
  nh.getParam("/max_ang_FOV", max_FOV);
  nh.getParam("/min_ang_FOV", min_FOV);
  nh.getParam("/pcTopic", pcTopic);
  nh.getParam("/imgTopic", imgTopic);

  nh.getParam("/x_resolution", angular_resolution_x);
  nh.getParam("/y_interpolation", interpol_value);

  nh.getParam("/ang_Y_resolution", angular_resolution_y);

  XmlRpc::XmlRpcValue param;

  // clang-format off
  nh.getParam("/matrix_file/tlc", param);
  Tlc << static_cast<double>(param[0]), static_cast<double>(param[1]), static_cast<double>(param[2]);

  nh.getParam("/matrix_file/rlc", param);

  Rlc << static_cast<double>(param[0]), static_cast<double>(param[1]), static_cast<double>(param[2]),
         static_cast<double>(param[3]), static_cast<double>(param[4]), static_cast<double>(param[5]),
         static_cast<double>(param[6]), static_cast<double>(param[7]), static_cast<double>(param[8]);

  nh.getParam("/matrix_file/camera_matrix", param);

  Mc << static_cast<double>(param[0]), static_cast<double>(param[1]), static_cast<double>(param[2]), static_cast<double>(param[3]),
        static_cast<double>(param[4]), static_cast<double>(param[5]), static_cast<double>(param[6]), static_cast<double>(param[7]),
        static_cast<double>(param[8]), static_cast<double>(param[9]), static_cast<double>(param[10]), static_cast<double>(param[11]);
  // clang-format on

  message_filters::Subscriber<PointCloud2> pc_sub(nh, pcTopic, 1);
  message_filters::Subscriber<Image> img_sub(nh, imgTopic, 1);

  using MySyncPolicy = sync_policies::ApproximateTime<PointCloud2, Image>;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), pc_sub, img_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  pcOnimg_pub = nh.advertise<sensor_msgs::Image>("/pcOnImage_image", 1);
  rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);

  pc_pub = nh.advertise<PointCloud>("/points2", 1);

  ros::spin();
}
