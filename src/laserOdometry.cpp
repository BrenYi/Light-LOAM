// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>
#include "omp.h"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0

int N_SCANS = 0;
int corner_correspondence = 0, plane_correspondence = 0;

constexpr double SCAN_PERIOD = 0.1;
constexpr double DISTANCE_SQ_THRESHOLD = 25;
constexpr double NEARBY_SCAN = 2.5;

int skipFrameNum = 5;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
std::mutex mBuf;



// undistort lidar point
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    if (DISTORTION)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame

void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

//new add
float Distance(pcl::PointXYZI& a, pcl::PointXYZI& b)
{
    float distance = -1;
    float dx, dy, dz;
    dx = a.x - b.x;
    dy = a.y - b.y;
    dz = a.z - b.z;
    distance = std::sqrt(dx*dx + dy*dy +dz*dz);
    return distance;
}


void graph_based_correspondence_vote_simple(std::vector<Corre_Match> &correspondences, bool corner_case, 
                    std::vector<Vertex_Vote> &selected_idx, 
                    Eigen::MatrixXi &graph_indexing, 
                    std::vector<Eigen::Vector3d> &feature_cur_point, 
                    std::vector<Eigen::Vector3d> &feature_last_point_a, 
                    std::vector<Eigen::Vector3d> &feature_last_point_b, 
                    std::vector<Eigen::Vector3d> &feature_last_point_c,
                    std::vector<double> &feature_s)
{
    
    int cor_size_all = correspondences.size();
    // int _horizontal_scans =2117;
    int number_of_region;
    float score_threshold;
    if(corner_case)
    {
        number_of_region = 5;
        score_threshold =0.96;
    }
    else
    {   
        score_threshold = 0.96;
        number_of_region = 10;
    }
    
        
        
        std::chrono::steady_clock::time_point tic_graph_correspondence = std::chrono::steady_clock::now();
        for(int num_region = 0;  num_region < number_of_region; num_region++)
        {
            std::vector<float> time_consuming;
            std::chrono::steady_clock::time_point tic_graph_loop8 = std::chrono::steady_clock::now();
            // std::cout << "num_region: " << num_region << std::endl;
            
            std::chrono::steady_clock::time_point tic_graph_construction = std::chrono::steady_clock::now();
            std::vector<Vertex_Attribute> vertex_set_param;
            
            int initial_pos = cor_size_all/number_of_region*(num_region);
            int end_pos;
            if(num_region == number_of_region-1)
            {
              end_pos = cor_size_all;
            }
            else
            {
              end_pos =  cor_size_all/number_of_region*(num_region+1);
            }
            std::vector<Corre_Match> correspondences_selected;
            correspondences_selected.reserve(cor_size_all/number_of_region*2);
            correspondences_selected.assign(correspondences.begin()+initial_pos, correspondences.begin()+end_pos);
            int cor_size = correspondences_selected.size();
            // std::cout << "sub cor size:" << correspondences_selected.size()<<std::endl;
            Eigen::MatrixXf compatibility_matrix;
            // int size = correspondences.size();
            compatibility_matrix.resize(correspondences_selected.size(), correspondences_selected.size());
            compatibility_matrix.setZero();
            // std::cout<< std::fixed<< std::setprecision(2);
            //test
            // Eigen::MatrixXf compatibility_matrix_t;
            // int size = correspondences.size();
            // compatibility_matrix_t.resize(correspondence_partial.size(), correspondence_partial.size());
            // compatibility_matrix_t.setZero();
            //test
            Corre_Match c1, c2;
            float s1, s2, dis_gap, resolution, alpha_param, score;
            resolution = 1; //_ang_resolution_X;
            // alpha_param =  resolution; // hyper parameter
            // std::vector<std::vector<float>> test_v(cor_size);
            std::vector<Vertex_Vote> vote_record(correspondences_selected.size(),{0,0.0});
            // std::vector<int> test(correspondences_selected.size(),0);
            // vote_record.reserve(correspondences_selected.size());
            for(int i = 0; i < correspondences_selected.size(); i++)
            {
                int count =0;
                vote_record[i].index = i;
                c1 = correspondences_selected[i];
                for(int j = i + 1; j < correspondences_selected.size(); j++)
                {
                    c2 = correspondences_selected[j];
                    s1 = Distance(c1.src, c2.src);
                    s2 = Distance(c1.tgt, c2.tgt);
                    dis_gap = std::abs(s1 - s2);
                    // float distance  = std::sqrt(c1.src.x*c1.src.x + c1.src.y*c1.src.y + c1.src.z*c1.src.z);
                    // resolution = 2 * distance * sin(_ang_resolution_X/2);
                    // std::cout<< "resolution:"<<resolution<<std::endl;
                    score = std::exp(-(dis_gap*dis_gap) / (resolution*resolution));
                    // if(score> 0.995) {count++;}
                    // score = (score < check_threshold) ? 0: score; //consider this way to filter more correspondence pairs to decrease compute load 
                    compatibility_matrix(i, j) = score;
                    compatibility_matrix(j, i) = score;
                    if(score < score_threshold)
                    {
                        count++;
                        vote_record[j].score += 1;
                        vote_record[i].score += 1;
                        // test[i] +=1;
                        // test[j] +=1;
                    }
                    // test_v[i].push_back(score);
                    // if(i == 1)
                    // {
                    //     test_v[0].push_back(score);
                    // }else if( i==10)
                    // {
                    //     test_v[1].push_back(score);
                    // }else if( i==15)
                    // {
                    //     test_v[2].push_back(score);
                    // }
                }
                // std::cout<<"sample:" << i<< " , count:"<< count<<" , percent:"<< count*1.0/correspondence_partial.size()*100<<"%"<<std::endl;
            }

            
            std::sort(vote_record.begin(), vote_record.end(), compare_score());
            int count_selected=0;
            if(corner_case)
            {
              
                float  selected_ratio = 0.90;
                float num_selected = selected_ratio*cor_size;
                float selected_count_ratio = 1;
                int donot_num_selected = (1-selected_count_ratio)*cor_size;
                // std::cout << "-------------------------------surf sub cor size:" << correspondences_selected.size()<<"--------------------------"<<std::endl;

                for(int i =cor_size-1; i >=0; i--)
                {
                    if(i >= donot_num_selected)
                    {   
                        Vertex_Vote obj;
                        // int indx = correspondences_selected[voter_ordered[i].index].index;
                        if(correspondences_selected[vote_record[i].index].index > correspondences.size()) continue;
                        obj.index = correspondences_selected[vote_record[i].index].index;
                        
                        if(vote_record[i].score > num_selected)
                        {

                            obj.score =0;
                            break;
                        }else if(vote_record[i].score  <=50){
                            // obj.score =0.2*vote_record[0].score/vote_record[i].score;
                            obj.score =9.0;
                        }else
                        {
                            obj.score =1;
                        }
                        
                        
                       
                        selected_idx.push_back(obj);count_selected++;
                        
                    }
                }
              
              
            }
            else
            {
                float  selected_ratio = 0.90;
                float num_selected = selected_ratio*cor_size;
                float selected_count_ratio = 1;
                int donot_num_selected = (1-selected_count_ratio)*cor_size;
                // std::cout << "---------------------------------sub cor size:" << correspondences_selected.size()<<"--------------------------"<<std::endl;

                for(int i =cor_size-1; i >=0; i--)
                {
                    if(i >= donot_num_selected)
                    {   
                        Vertex_Vote obj;
                        // int indx = correspondences_selected[voter_ordered[i].index].index;
                        if(correspondences_selected[vote_record[i].index].index > correspondences.size()) continue;
                        obj.index = correspondences_selected[vote_record[i].index].index;
                        
                        if(vote_record[i].score > num_selected)
                        {

                            obj.score =0;
                            break;
                        }else if(vote_record[i].score  <=50){
                            // obj.score =0.2*vote_record[0].score/vote_record[i].score;
                            obj.score =9.0;
                        }else
                        {
                            obj.score =1;
                        }
                        
                   
                       
                        selected_idx.push_back(obj);count_selected++;
                        
                    }
                }
            }
            vote_record.clear();
            vote_record.shrink_to_fit();



         }

        std::chrono::steady_clock::time_point toc_graph_correspondence = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used_init_graph_correspondence = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_correspondence - tic_graph_correspondence);


}

//new add
int main(int argc, char **argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
    nh.param<int>("scan_line", N_SCANS, 16);
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    //printf("Mapping %d Hz \n", 10 / skipFrameNum);

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    nav_msgs::Path laserPath;

    int frameCount = 0;
    int now_frame =0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        ros::spinOnce();

        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                //printf("unsync messeage!");
                ROS_BREAK();
            }

            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();

            TicToc t_whole;
            // initializing
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();
                std::cout<< "cornerPointsSharpNum:"<<cornerPointsSharpNum<<std::endl;
                std::cout<< "surfPointsFlatNum   :   "<<surfPointsFlatNum<<std::endl;
                TicToc t_opt;
                for (size_t opti_counter = 0; opti_counter < 3; ++opti_counter)
                {
                    //new add
                    int intilize_num = (cornerPointsSharpNum > surfPointsFlatNum) ? cornerPointsSharpNum : surfPointsFlatNum;
                    std::vector<Eigen::Vector3d> feature_cur_point;
                    // std::cout<<"corner_cur_point_size:"<<feature_cur_point.size()<<" , corner_cur_point_capacity："<<feature_cur_point.capacity()<<std::endl;
                    std::vector<Eigen::Vector3d> feature_last_point_a;
                    std::vector<Eigen::Vector3d> feature_last_point_b;
                    std::vector<Eigen::Vector3d> feature_last_point_c;
                    std::vector<double> feature_s;
                    std::vector<Corre_Match> correspondences;
                    std::vector<Vertex_Vote> selected_idx;
                    feature_cur_point.reserve(intilize_num);
                    feature_last_point_a.reserve(intilize_num);
                    feature_last_point_b.reserve(intilize_num);
                    feature_last_point_c.reserve(intilize_num);
                    feature_s.reserve(intilize_num);
                    correspondences.reserve(intilize_num);
                    selected_idx.reserve(intilize_num);
                    int index=0;
                    int _horizontal_scans =2117;
                    float _ang_resolution_X = (M_PI*2) / (_horizontal_scans);;
                    Eigen::MatrixXi graph_indexing;
                    graph_indexing.setOnes(N_SCANS, _horizontal_scans);
                    graph_indexing = graph_indexing * -1;
                    // float min_location;
                    float vertical_angle_top = 2;
                    float _ang_bottom=-24.9;
                    float _ang_resolution_Y = M_PI / 180.0 *(vertical_angle_top - _ang_bottom) / float(N_SCANS-1);
                    _ang_bottom = -( _ang_bottom - 0.1) * M_PI / 180.0;
                    // new add
                    
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    TicToc t_data;
                    // find correspondence for corner features
                    
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;

                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);

                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }

                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);
                            

                            double s;
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            feature_cur_point.push_back(curr_point);
                            feature_last_point_a.push_back(last_point_a);
                            feature_last_point_b.push_back(last_point_b);
                            feature_s.push_back(s);
                            // new add
                            Corre_Match cor;
                            cor.index = index;
                            cor.src = cornerPointsSharp->points[i];
                            cor.tgt = laserCloudCornerLast->points[closestPointInd];
                            cor.score = 0;
                            cor.s = 1/s;
                            correspondences.push_back(cor);
                            // cor.tgt = 
                            // std::cout<<"i:" <<i<<"  , cornerPointsSharp->points[i] col:"<< cornerlocation_mark[i]<< std::endl;
                            // std::cout<< "cornerPointsSharp->points[i] col:"<< int((cornerlocation_mark[i]-min_location)*_horizontal_scans<< std::endl;
                            //block the point
                            // PointType thisPoint = cornerPointsSharp->points[i];
                            // float range = sqrt(thisPoint.x * thisPoint.x +
                            //             thisPoint.y * thisPoint.y +
                            //             thisPoint.z * thisPoint.z);
                            // float verticalAngle = std::asin(thisPoint.z / range);
                            // int rowIdn = (verticalAngle + _ang_bottom) / _ang_resolution_Y;
                            // float horizonAngle = std::atan2(thisPoint.x, thisPoint.y);
                            // int columnIdn = -round((horizonAngle - M_PI_2) / _ang_resolution_X) + _horizontal_scans * 0.5;

                            // if (columnIdn >= _horizontal_scans){
                            // columnIdn -= _horizontal_scans;
                            // }

                            // // if (columnIdn < 0 || columnIdn >= _horizontal_scans){
                            // //   continue;
                            // // }

                            // graph_indexing(rowIdn, columnIdn) = index;
                            index++;
                            // if(now_frame <= 5)
                            // {
                            //     ceres::CostFunction *cost_function = LidarEdgeFactor_modify::Create(curr_point, last_point_a, last_point_b, s, 1);
                            //     problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            
                            // }
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }

                    }
                    // std::cout<<"corner_correspondence:" <<corner_correspondence<<std::endl;
                    // graph_based_correspondence_vote_partial(correspondences, true, selected_idx, graph_indexing);
                //    if(frameCount < 3)
                //    {
                //        graph_based_correspondence_vote_simple(correspondences, true, selected_idx, graph_indexing);
                //    }
                    // std::cout<< "filter_size_remain:"<<selected_idx.size()<<std::endl;
                    // if(now_frame > 5  )
                    // {
                    //     graph_based_correspondence_vote_simple(correspondences, true, selected_idx, graph_indexing,feature_cur_point, feature_last_point_a, feature_last_point_b, feature_last_point_c, feature_s);
                    //     for(int i = 0; i < selected_idx.size(); i++)
                    //     {
                        
                    //     int idx = selected_idx[i].index;
                    //     // std::cout<<"feature_cur_point[idx]:"<<idx<<std::endl;
                    //     // ceres::CostFunction *cost_function = LidarEdgeFactor::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx], feature_s[idx]);
                    //     // problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    //     ceres::CostFunction *cost_function = LidarEdgeFactor_modify::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx], feature_s[idx], selected_idx[i].score);
                    //     problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);

                    //     }

                    // }
                    selected_idx.clear();
                    correspondences.clear();
                    feature_cur_point.clear();
                    feature_last_point_a.clear();
                    feature_last_point_b.clear();
                    feature_s.clear();
                    index =0;

                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];

                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;

                                feature_cur_point.push_back(curr_point);
                                feature_last_point_a.push_back(last_point_a);
                                feature_last_point_b.push_back(last_point_b);
                                feature_last_point_c.push_back(last_point_c);
                                feature_s.push_back(s);
                                // new add
                                Corre_Match cor;
                                cor.index = index;
                                cor.src = surfPointsFlat->points[i];
                                cor.tgt = laserCloudSurfLast->points[closestPointInd];
                                cor.score = 0;
                                cor.s = 1/s;
                                correspondences.push_back(cor);
                                // cor.tgt = 
                                // std::cout<<"i:" <<i<<"  , cornerPointsSharp->points[i] col:"<< cornerlocation_mark[i]<< std::endl;
                                // std::cout<< "cornerPointsSharp->points[i] col:"<< int((cornerlocation_mark[i]-min_location)*_horizontal_scans<< std::endl;
                                //block the point
                                // PointType thisPoint = cornerPointsSharp->points[i];
                                // float range = sqrt(thisPoint.x * thisPoint.x +
                                //             thisPoint.y * thisPoint.y +
                                //             thisPoint.z * thisPoint.z);
                                // float verticalAngle = std::asin(thisPoint.z / range);
                                // int rowIdn = (verticalAngle + _ang_bottom) / _ang_resolution_Y;
                                // float horizonAngle = std::atan2(thisPoint.x, thisPoint.y);
                                // int columnIdn = -round((horizonAngle - M_PI_2) / _ang_resolution_X) + _horizontal_scans * 0.5;

                                // if (columnIdn >= _horizontal_scans){
                                // columnIdn -= _horizontal_scans;
                                // }

                                // // if (columnIdn < 0 || columnIdn >= _horizontal_scans){
                                // //   continue;
                                // // }

                                // graph_indexing(rowIdn, columnIdn) = index;
                                index++;
                                if(now_frame <= 5)  //ori:20
                                {
                                    ceres::CostFunction *cost_function = LidarPlaneFactor_modify::Create(curr_point, last_point_a, last_point_b, last_point_c, s, 1);
                                    // ceres::CostFunction *cost_function = LidarPlaneFactor_modify_test::Create(curr_point, last_point_a, last_point_b, last_point_c, s, 1);
                                    problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                
                                }
                                // ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                // problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }
                    if(now_frame > 5  )
                    {
                        graph_based_correspondence_vote_simple(correspondences, false, selected_idx, graph_indexing, feature_cur_point, feature_last_point_a, feature_last_point_b, feature_last_point_c, feature_s);
                        for(int i = 0; i < selected_idx.size(); i++)
                        {
                        
                        int idx = selected_idx[i].index;
                        // std::cout<<"feature_cur_point[idx]:"<<idx<<std::endl;
                        // ceres::CostFunction *cost_function = LidarEdgeFactor::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx], feature_s[idx]);
                        // problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                        ceres::CostFunction *cost_function = LidarPlaneFactor_modify::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx], feature_last_point_c[idx], feature_s[idx], selected_idx[i].score);
                        // ceres::CostFunction *cost_function = LidarPlaneFactor_modify_test::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx], feature_last_point_c[idx], feature_s[idx], selected_idx[i].score);
                        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);

                        }

                    }
                    ////printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    //printf("data association time %f ms \n", t_data.toc());

                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        //printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;
                    ceres::Solver::Options options;
                    options.linear_solver_type = ceres::DENSE_QR;
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    //printf("solver time %f ms \n", t_solver.toc());
                }
                //printf("optimization twice time %f \n", t_opt.toc());

                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "/camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();

            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "/camera_init";
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            if (0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            //printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
            now_frame++;
        }
        rate.sleep();
    }
    return 0;
}