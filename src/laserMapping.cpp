#include <math.h>
#include <vector>
#include <aloam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <fstream>


#include "lidarFactor.hpp"
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

bool init_flag = true;
Eigen::Matrix4f H_init;
Eigen::Matrix4f H;
std::string RESULT_PATH;


int frameCount = 0;

double timeLaserCloudCornerLast = 0;
double timeLaserCloudSurfLast = 0;
double timeLaserCloudFullRes = 0;
double timeLaserOdometry = 0;


int laserCloudCenWidth = 10;
int laserCloudCenHeight = 10;
int laserCloudCenDepth = 5;
const int laserCloudWidth = 21;
const int laserCloudHeight = 21;
const int laserCloudDepth = 11;


const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth; //4851


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];

// input: from odom
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

// ouput: all visualble cube points
pcl::PointCloud<PointType>::Ptr laserCloudSurround(new pcl::PointCloud<PointType>());

// surround points in map to build tree
pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<PointType>());

//input & output: points in one frame. local --> global
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// points in every cube
pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];

//kd-tree
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<PointType>());

double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
Eigen::Map<Eigen::Quaterniond> q_w_curr(parameters);
Eigen::Map<Eigen::Vector3d> t_w_curr(parameters + 4);

// wmap_T_odom * odom_T_curr = wmap_T_curr;
// transformation between odom's world and map's world frame
Eigen::Quaterniond q_wmap_wodom(1, 0, 0, 0);
Eigen::Vector3d t_wmap_wodom(0, 0, 0);

Eigen::Quaterniond q_wodom_curr(1, 0, 0, 0);
Eigen::Vector3d t_wodom_curr(0, 0, 0);


std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLastBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::mutex mBuf;

pcl::VoxelGrid<PointType> downSizeFilterCorner;
pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;

PointType pointOri, pointSel;

ros::Publisher pubLaserCloudSurround, pubLaserCloudMap, pubLaserCloudFullRes, pubOdomAftMapped, pubOdomAftMappedHighFrec, pubLaserAfterMappedPath;

nav_msgs::Path laserAfterMappedPath;

// set initial guess
void transformAssociateToMap()
{
	q_w_curr = q_wmap_wodom * q_wodom_curr;
	t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom;
}

void transformUpdate()
{
	q_wmap_wodom = q_w_curr * q_wodom_curr.inverse();
	t_wmap_wodom = t_w_curr - q_wmap_wodom * t_wodom_curr;
}

void pointAssociateToMap(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
	po->x = point_w.x();
	po->y = point_w.y();
	po->z = point_w.z();
	po->intensity = pi->intensity;
	//po->intensity = 1.0;
}

void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
{
	Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
	Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
	po->x = point_curr.x();
	po->y = point_curr.y();
	po->z = point_curr.z();
	po->intensity = pi->intensity;
}

void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	mBuf.lock();
	cornerLastBuf.push(laserCloudCornerLast2);
	mBuf.unlock();
}

void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	mBuf.lock();
	surfLastBuf.push(laserCloudSurfLast2);
	mBuf.unlock();
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	mBuf.lock();
	fullResBuf.push(laserCloudFullRes2);
	mBuf.unlock();
}

//receive odomtry
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(laserOdometry);
	mBuf.unlock();

	// high frequence publish
	Eigen::Quaterniond q_wodom_curr;
	Eigen::Vector3d t_wodom_curr;
	q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
	q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
	q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
	q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
	t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
	t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
	t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

	Eigen::Quaterniond q_w_curr = q_wmap_wodom * q_wodom_curr;
	Eigen::Vector3d t_w_curr = q_wmap_wodom * t_wodom_curr + t_wmap_wodom; 

	// correct the rotation difference
			//roll
			double sinr_cosp = 2 * (q_w_curr.w() * q_w_curr.x() + q_w_curr.y()*q_w_curr.z());
			double cosr_cosp = 1 - 2 * (q_w_curr.x() * q_w_curr.x() + q_w_curr.y() * q_w_curr.y());
    		double roll = std::atan2(sinr_cosp, cosr_cosp);
 
    		// pitch (y-axis rotation)
    		double sinp = 2 * (q_w_curr.w() * q_w_curr.y() - q_w_curr.z() * q_w_curr.x());
    		double pitch;
			if (std::abs(sinp) >= 1)
        		pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    		else
        		pitch = std::asin(sinp);
 
			// yaw (z-axis rotation)
			double siny_cosp = 2 * (q_w_curr.w() * q_w_curr.z() + q_w_curr.x() * q_w_curr.y());
			double cosy_cosp = 1 - 2 * (q_w_curr.y() * q_w_curr.y() + q_w_curr.z() * q_w_curr.z());
			double yaw = std::atan2(siny_cosp, cosy_cosp);
			// double w_new = q_w_curr.w();
			// double x_new = q_w_curr.x();
			// double y_new = q_w_curr.y();
			// double z_new = q_w_curr.z();

			roll = roll + M_PI / 2;
			yaw = yaw + M_PI / 2;


			double cy = cos(yaw * 0.5);
			double sy = sin(yaw * 0.5);
			double cp = cos(pitch * 0.5);
			double sp = sin(pitch * 0.5);
			double cr = cos(roll * 0.5);
			double sr = sin(roll * 0.5);
		
			Eigen::Quaterniond q_after;
			q_after.w() = cy * cp * cr + sy * sp * sr;
			q_after.x() = cy * cp * sr - sy * sp * cr;
			q_after.y() = sy * cp * sr + cy * sp * cr;
			q_after.z() = sy * cp * cr - cy * sp * sr;


			
	nav_msgs::Odometry odomAftMapped;
	odomAftMapped.header.frame_id = "/camera_init";
	odomAftMapped.child_frame_id = "/aft_mapped";
	odomAftMapped.header.stamp = laserOdometry->header.stamp;
	// odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
	// odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
	// odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
	// odomAftMapped.pose.pose.orientation.w = q_w_curr.w();

	odomAftMapped.pose.pose.orientation.x = q_after.y();
	odomAftMapped.pose.pose.orientation.y = -q_after.x();
	odomAftMapped.pose.pose.orientation.z = q_after.w();
	odomAftMapped.pose.pose.orientation.w = -q_after.z();
	odomAftMapped.pose.pose.position.x = t_w_curr.x();
	odomAftMapped.pose.pose.position.y = t_w_curr.y();
	odomAftMapped.pose.pose.position.z = t_w_curr.z();
	pubOdomAftMappedHighFrec.publish(odomAftMapped);
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

Eigen::MatrixXf graph_construction_partial(std::vector<Corre_Match> &correspondence_partial)
{
    Eigen::MatrixXf compatibility_matrix;
    // int size = correspondences.size();
    compatibility_matrix.resize(correspondence_partial.size(), correspondence_partial.size());
    compatibility_matrix.setZero();
    std::cout<< std::fixed<< std::setprecision(2);
    //test
    Eigen::MatrixXf compatibility_matrix_t;
    // int size = correspondences.size();
    // compatibility_matrix_t.resize(correspondence_partial.size(), correspondence_partial.size());
    // compatibility_matrix_t.setZero();
    //test
    Corre_Match c1, c2;
    float s1, s2, dis_gap, resolution, alpha_param, score;
    resolution = 1; //_ang_resolution_X;
    // alpha_param =  resolution; // hyper parameter
    // std::vector<std::vector<float>> test_v(cor_size);
    
    for(int i = 0; i < correspondence_partial.size(); i++)
    {
      int count =0;
        c1 = correspondence_partial[i];
        for(int j = i + 1; j < correspondence_partial.size(); j++)
        {
            c2 = correspondence_partial[j];
            s1 = Distance(c1.src, c2.src);
            s2 = Distance(c1.tgt, c2.tgt);
            dis_gap = std::abs(s1 - s2);
            // float distance  = std::sqrt(c1.src.x*c1.src.x + c1.src.y*c1.src.y + c1.src.z*c1.src.z);
            // resolution = 2 * distance * sin(_ang_resolution_X/2);
            // std::cout<< "resolution:"<<resolution<<std::endl;
            score = std::exp(-(dis_gap*dis_gap) / (resolution*resolution));
            if(score> 0.995) {count++;}
            // score = (score < check_threshold) ? 0: score; //consider this way to filter more correspondence pairs to decrease compute load 
            compatibility_matrix(i, j) = score;
            compatibility_matrix(j, i) = score;
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
    // std::cout<< "Graph:"<< compatibility_matrix.row(1)<<std::endl;
    // std::cout<<"Graph_t:"<<compatibility_matrix_t.row(1)<<std::endl;
    // float av_0 = std::accumulate(test_v[0].begin(), test_v[0].end(), 0.0) / test_v[0].size();
    // float av_1 = std::accumulate(test_v[1].begin(), test_v[1].end(), 0.0) / test_v[1].size();
    // float av_2 = std::accumulate(test_v[2].begin(), test_v[2].end(), 0.0) / test_v[2].size();
    // std::vector<float> av({av_0,av_1,av_2});
    return compatibility_matrix;
}

void graph_based_correspondence_vote_partial(std::vector<Corre_Match> &correspondences, bool corner_case, std::vector<Vertex_Vote> &selected_idx, Eigen::MatrixXi &graph_indexing)
{
    
    int cor_size_all = correspondences.size();
    // int _horizontal_scans =2117;
    int number_of_region = 10;
    // if(cor_size_all >250)
    // {
    //      std::cout << "no graph map,   cosrrespondences sizes:" << cor_size_all <<std::endl;
    //     return;
    // }
    
    // if((intercount%5) == 0)
    // {
        
        // std::cout << "Odometry cosrrespondences sizes:" << cor_size_all <<std::endl;
        
        
        std::chrono::steady_clock::time_point tic_graph_correspondence = std::chrono::steady_clock::now();
        for(int num_region = 0;  num_region < number_of_region; num_region++)
        {
            std::vector<float> time_consuming;
            std::chrono::steady_clock::time_point tic_graph_loop8 = std::chrono::steady_clock::now();
            // std::cout << "num_region: " << num_region << std::endl;
            
            std::chrono::steady_clock::time_point tic_graph_construction = std::chrono::steady_clock::now();
            std::vector<Vertex_Attribute> vertex_set_param;
            // int initial_pos = _horizontal_scans/6*(num_region);
            // int end_pos =  _horizontal_scans/6*(num_region+1);
            
            // correspondences_selected.reserve(500);
            // int count_idx=0;
            // for(int row =0 ; row < N_SCANS; row++)
            // {
            //     for(int col =initial_pos; col < end_pos; col++ )
            //     {
            //       int index_set = graph_indexing(row, col);
            //         if(index_set != -1)
            //         {
            //             // int x = graph_indexing(row, _horizontal_scans);
            //             correspondences_selected.push_back(correspondences[index_set]);
            //             // printf("index:%d",graph_indexing(row, col));
            //             // graph_indexing(row, col) = -1;
            //             // printf("index_after:%d \n",graph_indexing(row, col));
            //             // count_idx ++;
            //         }
            //     }
            // }
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
            
            // std::cout << "count_idx:" << count_idx<<std::endl;
            // std::cout << "sub cor size:" << correspondences_selected.size()<<std::endl;
            // std::cout << "sub cor capacity:" << correspondences_selected.capacity()<<std::endl;
            Eigen::MatrixXf Graph = graph_construction_partial(correspondences_selected);
            std::chrono::steady_clock::time_point toc_graph_construction = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_construction = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_construction - tic_graph_construction);
            time_consuming.push_back(time_graph_construction.count()*1000);
            // std::cout<< "Graph:"<< Graph<<std::endl;
            if( Graph.norm() == 0)
            {
                std::cout << "Graph is not connected!"<< std::endl;
                continue; 
            }
            std::chrono::steady_clock::time_point tic_graph_loop1 = std::chrono::steady_clock::now();
            std::vector<int> degree(cor_size, 0);

            for(int i = 0 ; i < cor_size; i++)
            {
                int test_cnt = 0;
                Vertex_Attribute va;
                std::vector<int> connected_idx;
                
                for(int j = 0; j < cor_size; j++)
                {
                    if(i != j && Graph(i,j) > 0.95)  //consider if modify  
                    {
                        degree[i]++;
                        connected_idx.push_back(j);
                    }
                    // if(i != j && Graph(i,j) )
                    // {
                    //     test_cnt +=1;
                    // }
                    
                }
                // std::cout << "trimmed: "<<degree[i]<<std::endl;
                // std::cout << "before: " << test_cnt<<std::endl;
                va.vertex_index = i;
                va.degree = degree[i];
                va.connected_vertex_index = connected_idx;
                vertex_set_param.push_back(va);
            }
            std::chrono::steady_clock::time_point toc_graph_loop1 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop1 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop1 - tic_graph_loop1);
            time_consuming.push_back(time_graph_loop1.count()*1000);
            std::vector<Vertex_Vote> neighbor_voter;
            float filter_param_numerator_a = 0;
            float filter_param_denominator_a = 0;
            float filter_param_b = 0;
            // double cluster_sum

            omp_set_num_threads(8);
            std::chrono::steady_clock::time_point tic_graph_loop2 = std::chrono::steady_clock::now();
            for(int i=0; i < cor_size; i++)
            {
                Vertex_Vote ob;
                // float weight_s = 0;
                std::vector<float> weight_s(vertex_set_param[i].degree * (vertex_set_param[i].degree -1) * 0.5, 0);
// #pragma omp parallel
                // {
#pragma omp parallel for default(none) shared(vertex_set_param, Graph, weight_s, i)
                for(int j = 0; j < vertex_set_param[i].degree; j++)
                {
                    
                    int index_neighbor_a = vertex_set_param[i].connected_vertex_index[j];
                    int count_b =0;
                    for(int k = j+1; k < vertex_set_param[i].degree; k++)
                    {
                    
                        int index_neighbor_b = vertex_set_param[i].connected_vertex_index[k];
                        if(Graph(index_neighbor_a, index_neighbor_b))
                        {
// #pragma omp critical
                            weight_s[int(j*(2*vertex_set_param[i].degree-1-j)/2+count_b)] = std::pow(Graph(i, index_neighbor_a) * Graph(i, index_neighbor_b) * Graph(index_neighbor_a, index_neighbor_b), 1.0/3);
                            
                        }
                        count_b++;
                        //  printf("thread is %d, j=%d, j address is %p, k=%d, k address is %p, weight_s address is %p, count_b address is %p\n", 
                        //         omp_get_thread_num(), j, &j, k, &k, &weight_s, &count_b);
                    }
                    
                }
                // }

                if(vertex_set_param[i].degree > 1)
                {
                    // float numerator = weight_s;
                    float numerator = std::accumulate(weight_s.begin(), weight_s.end(), 0.0);
                    float denominator = vertex_set_param[i].degree * (vertex_set_param[i].degree -1) * 0.5;
                    float factor = numerator / denominator;
                    filter_param_numerator_a += numerator;
                    filter_param_denominator_a += denominator;
                    ob.index = i;
                    ob.score = numerator / denominator;
                    neighbor_voter.push_back(ob);
                    // printf("teste.....,,,");
                }
                else
                {
                    ob.index = i;
                    ob.score = 0;
                    neighbor_voter.push_back(ob);
                }
                filter_param_b += ob.score;

            }
            float filter_param_a = filter_param_numerator_a / filter_param_denominator_a;
            filter_param_b = filter_param_b / neighbor_voter.size(); 
            float threshold_param = std::min(filter_param_a, filter_param_b);


            std::chrono::steady_clock::time_point toc_graph_loop2 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop2 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop2 - tic_graph_loop2);
            time_consuming.push_back(time_graph_loop2.count()*1000);
            
            // std::cout << "threshold_param1:" << threshold_param1 << std::endl;
            // std::cout << "threshold_param:" << threshold_param << std::endl;

            std::chrono::steady_clock::time_point tic_graph_loop3 = std::chrono::steady_clock::now();
            // order the correspondences in descending order
            std::vector<Vertex_Attribute> vertex_set_param_ordered;
            std::vector<Vertex_Vote> neighbor_voter_ordered;
            // vertex_set_param_ordered.assign(vertex_set_param.begin(), vertex_set_param.end());
            // neighbor_voter_ordered.assign(neighbor_voter.begin(), neighbor_voter.end());
            // std::sort(vertex_set_param_ordered.begin(), vertex_set_param_ordered.end(), compare_degree);
            // std::sort(neighbor_voter_ordered.begin(), neighbor_voter_ordered.end(), compare_score);
            std::chrono::steady_clock::time_point toc_graph_loop3 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop3 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop3 - tic_graph_loop3);
            time_consuming.push_back(time_graph_loop3.count()*1000);
            // print the outcome about voting process
            // if(log_flag)
            // {
            //     std::string degree_file = folder_data_primary_path + "/degree.txt";
            //     std::string score_file = folder_data_primary_path + "/cluster.txt";
            //     FILE *exp = fopen(degree_file.c_str(), "w");
            //     for(int i=0; i < cor_size; i++)
            //     {
            //         fprintf(exp, "%d : %d\n", vertex_set_param_ordered[i].vertex_index, vertex_set_param_ordered[i].degree);

            //     }
            //     fclose(exp);
            //     exp = fopen(score_file.c_str(), "w");
            //     for(int i = 0; i < cor_size; i++)
            //     {
            //         fprintf(exp, "%d : %lf\n", neighbor_voter_ordered[i].index, neighbor_voter_ordered[i].score);
            //     }
            //     fclose(exp);
            // }
            std::chrono::steady_clock::time_point tic_graph_loop4 = std::chrono::steady_clock::now();

            for(int i =0; i< cor_size; i++)
            {
                //have some problem
                std::vector<int> index_pruned;
                int new_degree = 0;
                // std::cout << "onnected_vertex_index.size()_before:"<< vertex_set_param[i].connected_vertex_index.size()<< std::endl;
// #pragma omp parallel for default(none) shared(vertex_set_param, neighbor_voter, index_pruned, new_degree, i, threshold_param)
                for(int j = 0 ; j <  vertex_set_param[i].connected_vertex_index.size(); j++)
                {
                    
                    int idx = vertex_set_param[i].connected_vertex_index[j];
                    if(neighbor_voter[idx].score >= threshold_param)
                    {
                        index_pruned.push_back(idx);
                        new_degree++;
                    }
                }
                vertex_set_param[i].connected_vertex_index.clear();
                vertex_set_param[i].connected_vertex_index = index_pruned;
                vertex_set_param[i].degree = index_pruned.size();
                // std::cout << "onnected_vertex_index.size()_after:"<< vertex_set_param[i].connected_vertex_index.size()<< std::endl;
            }

            std::chrono::steady_clock::time_point toc_graph_loop4 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop4 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop4 - tic_graph_loop4);
            time_consuming.push_back(time_graph_loop4.count()*1000);
            //calculate a full score
            float weight_balance = 0.9;
            double score_avarage = 0.0;
            std::vector<Vertex_Vote> voter;
            float score_all = 0;
            // float looser_score = 0;
            // float tight_score = 0;
            float min_score = 9999;

            std::chrono::steady_clock::time_point tic_graph_loop5 = std::chrono::steady_clock::now();
            for(int i =0; i < cor_size; i++)
            {
                score_all = 0;
                std::vector<float> looser_score(vertex_set_param[i].connected_vertex_index.size(), 0);
                std::vector<float> tight_score(vertex_set_param[i].connected_vertex_index.size() * (vertex_set_param[i].connected_vertex_index.size() -1) * 0.5, 0);
                Vertex_Vote obj;
                float tight_score_sum = 0;
                float looser_score_sum = 0;
// #pragma omp parallel
                {
// #pragma omp for
                if(vertex_set_param[i].connected_vertex_index.size() > 2)
                {
#pragma omp parallel for default(none) shared(vertex_set_param, i, looser_score, Graph, tight_score)
                for(int j = 0 ; j < vertex_set_param[i].connected_vertex_index.size(); j++)
                {
                    
                    int idx_a = vertex_set_param[i].connected_vertex_index[j];
// #pragma omp critical
                    looser_score[j] =  Graph(idx_a, i);
                    int count_b = 0;
                    for(int k = j+1 ; k < vertex_set_param[i].connected_vertex_index.size(); k++)
                    {
                        int idx_b = vertex_set_param[i].connected_vertex_index[k];
                        if(Graph(idx_a, idx_b))
                        {
// #pragma omp critical
                            tight_score[int(j*(2*vertex_set_param[i].connected_vertex_index.size()-1-j)/2+count_b)] = std::pow(Graph(idx_a, idx_b) * Graph(idx_a, i)* Graph(idx_b, i), 1/3);
                            
                        }
                        count_b++;
                    }
                }
                tight_score_sum = std::accumulate(tight_score.begin(), tight_score.end(), 0.0);
                tight_score_sum /= (vertex_set_param[i].degree * (vertex_set_param[i].degree -2)/2);
                }
                }
                if(vertex_set_param[i].degree != 0)
                {
                    looser_score_sum = std::accumulate(looser_score.begin(), looser_score.end(), 0.0);
                    looser_score_sum = looser_score_sum / vertex_set_param[i].degree;
                }
                // else
                // {
                //     std::cout<< "looser_score:" <<looser_score <<" tight_score:" << tight_score<< std::endl;
                // }
                
                // tight_score = tight_score / (vertex_set_param[i].degree * (vertex_set_param[i].degree -2)/2);
                score_all = (1-weight_balance) * looser_score_sum + weight_balance * tight_score_sum;
                score_avarage += score_all;
                obj.index = i;
                obj.score = score_all;
                if(score_all < min_score && score_all !=0) min_score = score_all;
                voter.push_back(obj);
            }
            std::vector<Vertex_Vote> voter_ordered;
            voter_ordered.assign(voter.begin(), voter.end());
            std::sort(voter_ordered.begin(), voter_ordered.end(), compare_score());
            score_avarage /=  cor_size;
            std::chrono::steady_clock::time_point toc_graph_loop5 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop5 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop5 - tic_graph_loop5);
            time_consuming.push_back(time_graph_loop5.count()*1000);
            std::chrono::steady_clock::time_point tic_graph_loop6 = std::chrono::steady_clock::now();

            int count_selected = 0;
            
            if(corner_case)
            {
              float  selected_ratio = 1;
              int num_selected = selected_ratio*cor_size;
              float max_score = voter_ordered[0].score;
              
              for(int i =0; i < cor_size; i++)
              {
                // mid = (begin + last)/2; 
                // guess = voter_ordered[mid];
                // if(voter_ordered[i].score > score_avarage)
                // {   
                //     Vertex_Vote obj;
                //     // int indx = correspondences_selected[voter_ordered[i].index].index;
                //     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
                //     obj.index = correspondences_selected[voter_ordered[i].index].index;
                //     obj.score = voter_ordered[i].score;
                //     selected_idx.push_back(obj);
                //     count_selected++;
                // }
                // else
                // {
                //     // std::cout << "i:" << i <<std::endl;
                //     // std::cout << "count_selected:" << count_selected <<std::endl;
                //     break;
                // }

                // select by ratio
                
                if(i < num_selected)
                {   
                    Vertex_Vote obj;
                    // int indx = correspondences_selected[voter_ordered[i].index].index;
                    if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
                    obj.index = correspondences_selected[voter_ordered[i].index].index;
                    obj.score =  voter_ordered[i].score;
                    // if(voter_ordered[i].score !=0)
                    // {
                    //     obj.score = (voter_ordered[i].score- min_score) / (max_score - min_score);
                    // }
                    // else
                    // {
                    //   obj.score = voter_ordered[i].score;
                    // }
                    
                    // std::cout<< "selected_id:" <<obj.index<<", score"<<obj.score<<std::endl;
                    if(obj.score !=0)
                    {selected_idx.push_back(obj);count_selected++;}
                    // if(obj.score !=0)
                    // {selected_idx.push_back(obj);count_selected++;}
                    // selected_idx.push_back(obj);count_selected++;
                    
                }
                else
                {
                    // std::cout << "i:" << i <<std::endl;
                    // std::cout << "count_selected:" << count_selected <<std::endl;
                    break;
                }
              }
            }
            else
            {
              float  selected_ratio = 1;
              int num_selected = selected_ratio*cor_size;
              for(int i =0; i < cor_size; i++)
              {
                // mid = (begin + last)/2; 
                // guess = voter_ordered[mid];
                // if(voter_ordered[i].score > score_avarage)
                // {   
                //     Vertex_Vote obj;
                //     // int indx = correspondences_selected[voter_ordered[i].index].index;
                //     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
                //     obj.index = correspondences_selected[voter_ordered[i].index].index;
                //     obj.score = voter_ordered[i].score;
                //     selected_idx.push_back(obj);
                //     count_selected++;
                // }
                // else
                // {
                //     // std::cout << "i:" << i <<std::endl;
                //     // std::cout << "count_selected:" << count_selected <<std::endl;
                //     break;
                // }

                // select by ratio
                
                if(i < num_selected)
                {   
                    Vertex_Vote obj;
                    // int indx = correspondences_selected[voter_ordered[i].index].index;
                    if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
                    obj.index = correspondences_selected[voter_ordered[i].index].index;
                    obj.score = voter_ordered[i].score;
                    // std::cout<< "selected_id:" <<obj.index<<", score"<<obj.score<<std::endl;
                    if(obj.score !=0)
                    {selected_idx.push_back(obj);count_selected++;}
                    // selected_idx.push_back(obj);count_selected++;
                    
                }
                else
                {
                    // std::cout << "i:" << i <<std::endl;
                    // std::cout << "count_selected:" << count_selected <<std::endl;
                    break;
                }
              }
            }
            std::chrono::steady_clock::time_point toc_graph_loop6 = std::chrono::steady_clock::now();
            float percent = count_selected*1.0 / cor_size;
            // if(int(percent*100) == 0 && cor_size!=0)
            
            if(count_selected == 0 && cor_size!=0)
            {
                // std::cout<< "error !!! Odometry!!!!  selected_size:"<<selected_idx.size()<<" , correspondences_selected_size:"<< cor_size<<std::endl;
              // std::cout << "cor_size:"<<cor_size<<std::endl;
              // std::cout << "selected_idx:"<<selected_idx.size()<<std::endl;
              if(corner_case)
              {
                std::cout<< "corner error !!! Odometry!!!!"<<std::endl;
              }
              else
              {
                std::cout<< "surf error !!! Odometry!!!!"<<std::endl;
              }
              
            //   for(int i=0; i< cor_size; i++)
            //   {
            //     Vertex_Vote obj;
            //     obj.index = correspondences_selected[i].index;
            //     obj.score = correspondences_selected[i].score;
            //     selected_idx.push_back(obj);
            //   }
              // selected_idx.insert(selected_idx.end(), correspondences_selected.cbegin(), correspondences_selected.cend());
              // selected_idx.append_range(correspondences_selected);
              // std::cout << "selected_idx_aft:"<<selected_idx.size()<<std::endl;
              // std::cout << "correspondences_selected:"<<correspondences_selected.size()<<std::endl;

            }
            // std::cout << setprecision(3) << setiosflags(ios::fixed);
            

            // visualization_correspondence(correspondences, selected_idx);
            // memcpy(visual_handle.transformCur , transformCur, sizeof(float)*6);
            // visual_handle.visualize_correspondence_relationships(correspondences, selected_idx);
        

            // std::cout << "Calculation Time showing:" <<std::endl;

            // std::cout << "Graph construction time: " << time_consuming[0]<<"ms" << std::endl;
            // std::cout << "Find connected  point: " << time_consuming[1]<< "ms" << std::endl;
            // std::cout << "Find minimal clique and calculate filter param: " << time_consuming[2]<< "ms" << std::endl;
            // std::cout << "Sort two vector: " << time_consuming[3]<< "ms" << std::endl;
            // std::cout << "Purne connected point set: " << time_consuming[4]<< "ms" << std::endl;
            // std::cout << "Tight and loose voter: " << time_consuming[5]<< "ms" << std::endl;
            // std::cout << "Sort and slect final correspondences: " << time_consuming[6]<< "ms" << std::endl;
            
            std::chrono::steady_clock::time_point tic_graph_loop7 = std::chrono::steady_clock::now();
            vertex_set_param.clear();
            vertex_set_param.shrink_to_fit();
            degree.clear();
            degree.shrink_to_fit();
            neighbor_voter.clear();
            neighbor_voter.shrink_to_fit();;
            vertex_set_param_ordered.clear();
            vertex_set_param_ordered.shrink_to_fit();
            neighbor_voter_ordered.clear();
            neighbor_voter_ordered.shrink_to_fit();
            voter_ordered.clear();
            voter_ordered.shrink_to_fit();
            voter.clear();
            voter.shrink_to_fit();
            std::chrono::steady_clock::time_point toc_graph_loop7 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop7 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop7 - tic_graph_loop7);
            time_consuming.push_back(time_graph_loop7.count()*1000);
            // std::cout << "shrink_to_fit and clear vectors: " << time_consuming[7]<< "ms" << std::endl;

            std::chrono::steady_clock::time_point toc_graph_loop8 = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_graph_loop8 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop8 - tic_graph_loop8);
            time_consuming.push_back(time_graph_loop8.count()*1000);
            // std::cout << "single sub graph loop consuming time: " << time_graph_loop8.count()*1000<< "ms" << std::endl;


        }
        // std::cout << "selected_idx(): " <<selected_idx.size() <<std::endl;
        // std::cout<< "selected_id:" <<selected_idx<<std::endl;
        std::cout << "Odom successfly slected by graph(percent): " << selected_idx.size()*1.0/cor_size_all * 100<<"%" <<std::endl;
    
        std::chrono::steady_clock::time_point toc_graph_correspondence = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used_init_graph_correspondence = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_correspondence - tic_graph_correspondence);
        std::cout << "Odom's Graph     corrrespondence     consuming    time      :" << time_used_init_graph_correspondence.count() *1000<< "ms"<< std::endl;
        
        
        
        
    // }

}

void graph_based_correspondence_vote_simple(std::vector<Corre_Match> &correspondences, bool corner_case, std::vector<Vertex_Vote> &selected_idx)
{
    
    int cor_size_all = correspondences.size();
    // int _horizontal_scans =2117;
    int number_of_region = 20;
    // if(cor_size_all >250)
    // {
    //      std::cout << "no graph map,   cosrrespondences sizes:" << cor_size_all <<std::endl;
    //     return;
    // }
    
    // if((intercount%5) == 0)
    // {
        
        // std::cout << "Odometry cosrrespondences sizes:" << cor_size_all <<std::endl;
        
        
        std::chrono::steady_clock::time_point tic_graph_correspondence = std::chrono::steady_clock::now();
        for(int num_region = 0;  num_region < number_of_region; num_region++)
        {
            std::vector<float> time_consuming;
            std::chrono::steady_clock::time_point tic_graph_loop8 = std::chrono::steady_clock::now();
            // std::cout << "num_region: " << num_region << std::endl;
            
            std::chrono::steady_clock::time_point tic_graph_construction = std::chrono::steady_clock::now();
            std::vector<Vertex_Attribute> vertex_set_param;
            // int initial_pos = _horizontal_scans/6*(num_region);
            // int end_pos =  _horizontal_scans/6*(num_region+1);
            
            // correspondences_selected.reserve(500);
            // int count_idx=0;
            // for(int row =0 ; row < N_SCANS; row++)
            // {
            //     for(int col =initial_pos; col < end_pos; col++ )
            //     {
            //       int index_set = graph_indexing(row, col);
            //         if(index_set != -1)
            //         {
            //             // int x = graph_indexing(row, _horizontal_scans);
            //             correspondences_selected.push_back(correspondences[index_set]);
            //             // printf("index:%d",graph_indexing(row, col));
            //             // graph_indexing(row, col) = -1;
            //             // printf("index_after:%d \n",graph_indexing(row, col));
            //             // count_idx ++;
            //         }
            //     }
            // }
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
            std::vector<int> test(correspondences_selected.size(),0);
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
                    if(score < 0.95)
                    {
                        count++;
                        vote_record[j].score += 1;
                        vote_record[i].score += 1;
                        test[i] +=1;
                        test[j] +=1;
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

            // for(int i =0; i<correspondences_selected.size(); i++)
            // {
            //     std::cout<<"index:"<< vote_record[i].index<< " , count:"<<vote_record[i].score<<std::endl;
            //     // std::cout<<"test_id:"<< i<< " , test_count:"<<test[i]<<std::endl;
            // }
            std::sort(vote_record.begin(), vote_record.end(), compare_score());
            int count_selected=0;
            if(corner_case)
            {
              float  selected_ratio = 0.75;
              float num_selected = selected_ratio*cor_size;
            //   std::cout<<"num_selected:"<<num_selected<<std::endl;
              for(int i =cor_size-1; i >=0; i--)
              {
                // mid = (begin + last)/2; 
                // guess = voter_ordered[mid];
                // if(voter_ordered[i].score > score_avarage)
                // {   
                //     Vertex_Vote obj;
                //     // int indx = correspondences_selected[voter_ordered[i].index].index;
                //     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
                //     obj.index = correspondences_selected[voter_ordered[i].index].index;
                //     obj.score = voter_ordered[i].score;
                //     selected_idx.push_back(obj);
                //     count_selected++;
                // }
                // else
                // {
                //     // std::cout << "i:" << i <<std::endl;
                //     // std::cout << "count_selected:" << count_selected <<std::endl;
                //     break;
                // }

                // select by ratio
                // std::cout<<"vote_record[i].score:"<<vote_record[i].score<<std::endl;
                // 
                if(vote_record[i].score < num_selected)
                {   
                    Vertex_Vote obj;
                    // int indx = correspondences_selected[voter_ordered[i].index].index;
                    if(correspondences_selected[vote_record[i].index].index > correspondences.size()) continue;
                    obj.index = correspondences_selected[vote_record[i].index].index;
                    obj.score =1.0;
                    
                    // if(voter_ordered[i].score !=0)
                    // {
                    //     obj.score = (voter_ordered[i].score- min_score) / (max_score - min_score);
                    // }
                    // else
                    // {
                    //   obj.score = voter_ordered[i].score;
                    // }
                    
                    // std::cout<< "selected_id:" <<vote_record[i].index<<" , score:"<<vote_record[i].score<<std::endl;
                    // if(obj.score !=0)
                    // {selected_idx.push_back(obj);count_selected++;}
                    // if(obj.score !=0)
                    // {selected_idx.push_back(obj);count_selected++;}
                    selected_idx.push_back(obj);count_selected++;
                    
                }
                else
                {
                    // std::cout << "i:" << i <<std::endl;
                    // std::cout << "count_selected:" << count_selected <<std::endl;
                    break;
                }
              }
            }
            vote_record.clear();
            vote_record.shrink_to_fit();

//             for(int i = 0; i < correspondences_selected.size(); i++)
//             {
//                 std::cout<< "correspondence pair:" << i<<" ,score: "<<vote_record[i]<<std::endl;
//             }
//             // std::cout<< "Graph:"<< vote_record<<std::endl;
//             // std::cout<<"Graph_t:"<<compatibility_matrix_t.row(1)<<std::endl;
//             // float av_0 = std::accumulate(test_v[0].begin(), test_v[0].end(), 0.0) / test_v[0].size();
//             // float av_1 = std::accumulate(test_v[1].begin(), test_v[1].end(), 0.0) / test_v[1].size();
//             // float av_2 = std::accumulate(test_v[2].begin(), test_v[2].end(), 0.0) / test_v[2].size();
//             // std::vector<float> av({av_0,av_1,av_2});

            
//             // std::cout << "count_idx:" << count_idx<<std::endl;
            
//             // std::cout << "sub cor capacity:" << correspondences_selected.capacity()<<std::endl;
//             Eigen::MatrixXf Graph = compatibility_matrix;
//             std::chrono::steady_clock::time_point toc_graph_construction = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_construction = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_construction - tic_graph_construction);
//             time_consuming.push_back(time_graph_construction.count()*1000);
//             // std::cout<< "Graph:"<< Graph<<std::endl;
//             if( Graph.norm() == 0)
//             {
//                 std::cout << "Graph is not connected!"<< std::endl;
//                 continue; 
//             }
//             std::chrono::steady_clock::time_point tic_graph_loop1 = std::chrono::steady_clock::now();
//             std::vector<int> degree(cor_size, 0);

//             for(int i = 0 ; i < cor_size; i++)
//             {
//                 int test_cnt = 0;
//                 Vertex_Attribute va;
//                 std::vector<int> connected_idx;
                
//                 for(int j = 0; j < cor_size; j++)
//                 {
//                     if(i != j && Graph(i,j) > 0.95)  //consider if modify  
//                     {
//                         degree[i]++;
//                         connected_idx.push_back(j);
//                     }
//                     // if(i != j && Graph(i,j) )
//                     // {
//                     //     test_cnt +=1;
//                     // }
                    
//                 }
//                 // std::cout << "trimmed: "<<degree[i]<<std::endl;
//                 // std::cout << "before: " << test_cnt<<std::endl;
//                 va.vertex_index = i;
//                 va.degree = degree[i];
//                 va.connected_vertex_index = connected_idx;
//                 vertex_set_param.push_back(va);
//             }
//             std::chrono::steady_clock::time_point toc_graph_loop1 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop1 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop1 - tic_graph_loop1);
//             time_consuming.push_back(time_graph_loop1.count()*1000);
//             std::vector<Vertex_Vote> neighbor_voter;
//             float filter_param_numerator_a = 0;
//             float filter_param_denominator_a = 0;
//             float filter_param_b = 0;
//             // double cluster_sum

//             omp_set_num_threads(8);
//             std::chrono::steady_clock::time_point tic_graph_loop2 = std::chrono::steady_clock::now();
//             for(int i=0; i < cor_size; i++)
//             {
//                 Vertex_Vote ob;
//                 // float weight_s = 0;
//                 std::vector<float> weight_s(vertex_set_param[i].degree * (vertex_set_param[i].degree -1) * 0.5, 0);
// // #pragma omp parallel
//                 // {
// #pragma omp parallel for default(none) shared(vertex_set_param, Graph, weight_s, i)
//                 for(int j = 0; j < vertex_set_param[i].degree; j++)
//                 {
                    
//                     int index_neighbor_a = vertex_set_param[i].connected_vertex_index[j];
//                     int count_b =0;
//                     for(int k = j+1; k < vertex_set_param[i].degree; k++)
//                     {
                    
//                         int index_neighbor_b = vertex_set_param[i].connected_vertex_index[k];
//                         if(Graph(index_neighbor_a, index_neighbor_b))
//                         {
// // #pragma omp critical
//                             weight_s[int(j*(2*vertex_set_param[i].degree-1-j)/2+count_b)] = std::pow(Graph(i, index_neighbor_a) * Graph(i, index_neighbor_b) * Graph(index_neighbor_a, index_neighbor_b), 1.0/3);
                            
//                         }
//                         count_b++;
//                         //  printf("thread is %d, j=%d, j address is %p, k=%d, k address is %p, weight_s address is %p, count_b address is %p\n", 
//                         //         omp_get_thread_num(), j, &j, k, &k, &weight_s, &count_b);
//                     }
                    
//                 }
//                 // }

//                 if(vertex_set_param[i].degree > 1)
//                 {
//                     // float numerator = weight_s;
//                     float numerator = std::accumulate(weight_s.begin(), weight_s.end(), 0.0);
//                     float denominator = vertex_set_param[i].degree * (vertex_set_param[i].degree -1) * 0.5;
//                     float factor = numerator / denominator;
//                     filter_param_numerator_a += numerator;
//                     filter_param_denominator_a += denominator;
//                     ob.index = i;
//                     ob.score = numerator / denominator;
//                     neighbor_voter.push_back(ob);
//                     // printf("teste.....,,,");
//                 }
//                 else
//                 {
//                     ob.index = i;
//                     ob.score = 0;
//                     neighbor_voter.push_back(ob);
//                 }
//                 filter_param_b += ob.score;

//             }
//             float filter_param_a = filter_param_numerator_a / filter_param_denominator_a;
//             filter_param_b = filter_param_b / neighbor_voter.size(); 
//             float threshold_param = std::min(filter_param_a, filter_param_b);


//             std::chrono::steady_clock::time_point toc_graph_loop2 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop2 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop2 - tic_graph_loop2);
//             time_consuming.push_back(time_graph_loop2.count()*1000);
            
//             // std::cout << "threshold_param1:" << threshold_param1 << std::endl;
//             // std::cout << "threshold_param:" << threshold_param << std::endl;

//             std::chrono::steady_clock::time_point tic_graph_loop3 = std::chrono::steady_clock::now();
//             // order the correspondences in descending order
//             std::vector<Vertex_Attribute> vertex_set_param_ordered;
//             std::vector<Vertex_Vote> neighbor_voter_ordered;
//             // vertex_set_param_ordered.assign(vertex_set_param.begin(), vertex_set_param.end());
//             // neighbor_voter_ordered.assign(neighbor_voter.begin(), neighbor_voter.end());
//             // std::sort(vertex_set_param_ordered.begin(), vertex_set_param_ordered.end(), compare_degree);
//             // std::sort(neighbor_voter_ordered.begin(), neighbor_voter_ordered.end(), compare_score);
//             std::chrono::steady_clock::time_point toc_graph_loop3 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop3 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop3 - tic_graph_loop3);
//             time_consuming.push_back(time_graph_loop3.count()*1000);
//             // print the outcome about voting process
//             // if(log_flag)
//             // {
//             //     std::string degree_file = folder_data_primary_path + "/degree.txt";
//             //     std::string score_file = folder_data_primary_path + "/cluster.txt";
//             //     FILE *exp = fopen(degree_file.c_str(), "w");
//             //     for(int i=0; i < cor_size; i++)
//             //     {
//             //         fprintf(exp, "%d : %d\n", vertex_set_param_ordered[i].vertex_index, vertex_set_param_ordered[i].degree);

//             //     }
//             //     fclose(exp);
//             //     exp = fopen(score_file.c_str(), "w");
//             //     for(int i = 0; i < cor_size; i++)
//             //     {
//             //         fprintf(exp, "%d : %lf\n", neighbor_voter_ordered[i].index, neighbor_voter_ordered[i].score);
//             //     }
//             //     fclose(exp);
//             // }
//             std::chrono::steady_clock::time_point tic_graph_loop4 = std::chrono::steady_clock::now();

//             for(int i =0; i< cor_size; i++)
//             {
//                 //have some problem
//                 std::vector<int> index_pruned;
//                 int new_degree = 0;
//                 // std::cout << "onnected_vertex_index.size()_before:"<< vertex_set_param[i].connected_vertex_index.size()<< std::endl;
// // #pragma omp parallel for default(none) shared(vertex_set_param, neighbor_voter, index_pruned, new_degree, i, threshold_param)
//                 for(int j = 0 ; j <  vertex_set_param[i].connected_vertex_index.size(); j++)
//                 {
                    
//                     int idx = vertex_set_param[i].connected_vertex_index[j];
//                     if(neighbor_voter[idx].score >= threshold_param)
//                     {
//                         index_pruned.push_back(idx);
//                         new_degree++;
//                     }
//                 }
//                 vertex_set_param[i].connected_vertex_index.clear();
//                 vertex_set_param[i].connected_vertex_index = index_pruned;
//                 vertex_set_param[i].degree = index_pruned.size();
//                 // std::cout << "onnected_vertex_index.size()_after:"<< vertex_set_param[i].connected_vertex_index.size()<< std::endl;
//             }

//             std::chrono::steady_clock::time_point toc_graph_loop4 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop4 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop4 - tic_graph_loop4);
//             time_consuming.push_back(time_graph_loop4.count()*1000);
//             //calculate a full score
//             float weight_balance = 0.9;
//             double score_avarage = 0.0;
//             std::vector<Vertex_Vote> voter;
//             float score_all = 0;
//             // float looser_score = 0;
//             // float tight_score = 0;
//             float min_score = 9999;

//             std::chrono::steady_clock::time_point tic_graph_loop5 = std::chrono::steady_clock::now();
//             for(int i =0; i < cor_size; i++)
//             {
//                 score_all = 0;
//                 std::vector<float> looser_score(vertex_set_param[i].connected_vertex_index.size(), 0);
//                 std::vector<float> tight_score(vertex_set_param[i].connected_vertex_index.size() * (vertex_set_param[i].connected_vertex_index.size() -1) * 0.5, 0);
//                 Vertex_Vote obj;
//                 float tight_score_sum = 0;
//                 float looser_score_sum = 0;
// // #pragma omp parallel
//                 {
// // #pragma omp for
//                 if(vertex_set_param[i].connected_vertex_index.size() > 2)
//                 {
// #pragma omp parallel for default(none) shared(vertex_set_param, i, looser_score, Graph, tight_score)
//                 for(int j = 0 ; j < vertex_set_param[i].connected_vertex_index.size(); j++)
//                 {
                    
//                     int idx_a = vertex_set_param[i].connected_vertex_index[j];
// // #pragma omp critical
//                     looser_score[j] =  Graph(idx_a, i);
//                     int count_b = 0;
//                     for(int k = j+1 ; k < vertex_set_param[i].connected_vertex_index.size(); k++)
//                     {
//                         int idx_b = vertex_set_param[i].connected_vertex_index[k];
//                         if(Graph(idx_a, idx_b))
//                         {
// // #pragma omp critical
//                             tight_score[int(j*(2*vertex_set_param[i].connected_vertex_index.size()-1-j)/2+count_b)] = std::pow(Graph(idx_a, idx_b) * Graph(idx_a, i)* Graph(idx_b, i), 1/3);
                            
//                         }
//                         count_b++;
//                     }
//                 }
//                 tight_score_sum = std::accumulate(tight_score.begin(), tight_score.end(), 0.0);
//                 tight_score_sum /= (vertex_set_param[i].degree * (vertex_set_param[i].degree -2)/2);
//                 }
//                 }
//                 if(vertex_set_param[i].degree != 0)
//                 {
//                     looser_score_sum = std::accumulate(looser_score.begin(), looser_score.end(), 0.0);
//                     looser_score_sum = looser_score_sum / vertex_set_param[i].degree;
//                 }
//                 // else
//                 // {
//                 //     std::cout<< "looser_score:" <<looser_score <<" tight_score:" << tight_score<< std::endl;
//                 // }
                
//                 // tight_score = tight_score / (vertex_set_param[i].degree * (vertex_set_param[i].degree -2)/2);
//                 score_all = (1-weight_balance) * looser_score_sum + weight_balance * tight_score_sum;
//                 score_avarage += score_all;
//                 obj.index = i;
//                 obj.score = score_all;
//                 if(score_all < min_score && score_all !=0) min_score = score_all;
//                 voter.push_back(obj);
//             }
//             std::vector<Vertex_Vote> voter_ordered;
//             voter_ordered.assign(voter.begin(), voter.end());
//             std::sort(voter_ordered.begin(), voter_ordered.end(), compare_score());
//             score_avarage /=  cor_size;
//             std::chrono::steady_clock::time_point toc_graph_loop5 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop5 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop5 - tic_graph_loop5);
//             time_consuming.push_back(time_graph_loop5.count()*1000);
//             std::chrono::steady_clock::time_point tic_graph_loop6 = std::chrono::steady_clock::now();

//             int count_selected = 0;
            
//             if(corner_case)
//             {
//               float  selected_ratio = 1;
//               int num_selected = selected_ratio*cor_size;
//               float max_score = voter_ordered[0].score;
              
//               for(int i =0; i < cor_size; i++)
//               {
//                 // mid = (begin + last)/2; 
//                 // guess = voter_ordered[mid];
//                 // if(voter_ordered[i].score > score_avarage)
//                 // {   
//                 //     Vertex_Vote obj;
//                 //     // int indx = correspondences_selected[voter_ordered[i].index].index;
//                 //     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
//                 //     obj.index = correspondences_selected[voter_ordered[i].index].index;
//                 //     obj.score = voter_ordered[i].score;
//                 //     selected_idx.push_back(obj);
//                 //     count_selected++;
//                 // }
//                 // else
//                 // {
//                 //     // std::cout << "i:" << i <<std::endl;
//                 //     // std::cout << "count_selected:" << count_selected <<std::endl;
//                 //     break;
//                 // }

//                 // select by ratio
                
//                 if(i < num_selected)
//                 {   
//                     Vertex_Vote obj;
//                     // int indx = correspondences_selected[voter_ordered[i].index].index;
//                     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
//                     obj.index = correspondences_selected[voter_ordered[i].index].index;
//                     if(obj.score != 0)
//                     {obj.score =  1.1*voter_ordered[i].score;}
//                     else
//                     {
//                         obj.score =1;
//                     }
//                     // if(voter_ordered[i].score !=0)
//                     // {
//                     //     obj.score = (voter_ordered[i].score- min_score) / (max_score - min_score);
//                     // }
//                     // else
//                     // {
//                     //   obj.score = voter_ordered[i].score;
//                     // }
                    
//                     std::cout<< "selected_id:" <<voter_ordered[i].index<<", score"<<voter_ordered[i].score<<std::endl;
//                     // if(obj.score !=0)
//                     // {selected_idx.push_back(obj);count_selected++;}
//                     // if(obj.score !=0)
//                     // {selected_idx.push_back(obj);count_selected++;}
//                     selected_idx.push_back(obj);count_selected++;
                    
//                 }
//                 else
//                 {
//                     // std::cout << "i:" << i <<std::endl;
//                     // std::cout << "count_selected:" << count_selected <<std::endl;
//                     break;
//                 }
//               }
//             }
//             else
//             {
//               float  selected_ratio = 1;
//               int num_selected = selected_ratio*cor_size;
//               for(int i =0; i < cor_size; i++)
//               {
//                 // mid = (begin + last)/2; 
//                 // guess = voter_ordered[mid];
//                 // if(voter_ordered[i].score > score_avarage)
//                 // {   
//                 //     Vertex_Vote obj;
//                 //     // int indx = correspondences_selected[voter_ordered[i].index].index;
//                 //     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
//                 //     obj.index = correspondences_selected[voter_ordered[i].index].index;
//                 //     obj.score = voter_ordered[i].score;
//                 //     selected_idx.push_back(obj);
//                 //     count_selected++;
//                 // }
//                 // else
//                 // {
//                 //     // std::cout << "i:" << i <<std::endl;
//                 //     // std::cout << "count_selected:" << count_selected <<std::endl;
//                 //     break;
//                 // }

//                 // select by ratio
                
//                 if(i < num_selected)
//                 {   
//                     Vertex_Vote obj;
//                     // int indx = correspondences_selected[voter_ordered[i].index].index;
//                     if(correspondences_selected[voter_ordered[i].index].index > correspondences.size()) continue;
//                     obj.index = correspondences_selected[voter_ordered[i].index].index;
//                     obj.score = voter_ordered[i].score;
//                     // std::cout<< "selected_id:" <<obj.index<<", score"<<obj.score<<std::endl;
//                     if(obj.score !=0)
//                     {selected_idx.push_back(obj);count_selected++;}
//                     // selected_idx.push_back(obj);count_selected++;
                    
//                 }
//                 else
//                 {
//                     // std::cout << "i:" << i <<std::endl;
//                     // std::cout << "count_selected:" << count_selected <<std::endl;
//                     break;
//                 }
//               }
//             }
//             std::chrono::steady_clock::time_point toc_graph_loop6 = std::chrono::steady_clock::now();
//             float percent = count_selected*1.0 / cor_size;
//             // if(int(percent*100) == 0 && cor_size!=0)
            
//             if(count_selected == 0 && cor_size!=0)
//             {
//                 // std::cout<< "error !!! Odometry!!!!  selected_size:"<<selected_idx.size()<<" , correspondences_selected_size:"<< cor_size<<std::endl;
//               // std::cout << "cor_size:"<<cor_size<<std::endl;
//               // std::cout << "selected_idx:"<<selected_idx.size()<<std::endl;
//               if(corner_case)
//               {
//                 std::cout<< "corner error !!! Odometry!!!!"<<std::endl;
//               }
//               else
//               {
//                 std::cout<< "surf error !!! Odometry!!!!"<<std::endl;
//               }
              
//             //   for(int i=0; i< cor_size; i++)
//             //   {
//             //     Vertex_Vote obj;
//             //     obj.index = correspondences_selected[i].index;
//             //     obj.score = correspondences_selected[i].score;
//             //     selected_idx.push_back(obj);
//             //   }
//               // selected_idx.insert(selected_idx.end(), correspondences_selected.cbegin(), correspondences_selected.cend());
//               // selected_idx.append_range(correspondences_selected);
//               // std::cout << "selected_idx_aft:"<<selected_idx.size()<<std::endl;
//               // std::cout << "correspondences_selected:"<<correspondences_selected.size()<<std::endl;

//             }
//             // std::cout << setprecision(3) << setiosflags(ios::fixed);
            

//             // visualization_correspondence(correspondences, selected_idx);
//             // memcpy(visual_handle.transformCur , transformCur, sizeof(float)*6);
//             // visual_handle.visualize_correspondence_relationships(correspondences, selected_idx);
        

//             // std::cout << "Calculation Time showing:" <<std::endl;

//             // std::cout << "Graph construction time: " << time_consuming[0]<<"ms" << std::endl;
//             // std::cout << "Find connected  point: " << time_consuming[1]<< "ms" << std::endl;
//             // std::cout << "Find minimal clique and calculate filter param: " << time_consuming[2]<< "ms" << std::endl;
//             // std::cout << "Sort two vector: " << time_consuming[3]<< "ms" << std::endl;
//             // std::cout << "Purne connected point set: " << time_consuming[4]<< "ms" << std::endl;
//             // std::cout << "Tight and loose voter: " << time_consuming[5]<< "ms" << std::endl;
//             // std::cout << "Sort and slect final correspondences: " << time_consuming[6]<< "ms" << std::endl;
            
//             std::chrono::steady_clock::time_point tic_graph_loop7 = std::chrono::steady_clock::now();
//             vertex_set_param.clear();
//             vertex_set_param.shrink_to_fit();
//             degree.clear();
//             degree.shrink_to_fit();
//             neighbor_voter.clear();
//             neighbor_voter.shrink_to_fit();;
//             vertex_set_param_ordered.clear();
//             vertex_set_param_ordered.shrink_to_fit();
//             neighbor_voter_ordered.clear();
//             neighbor_voter_ordered.shrink_to_fit();
//             voter_ordered.clear();
//             voter_ordered.shrink_to_fit();
//             voter.clear();
//             voter.shrink_to_fit();
//             std::chrono::steady_clock::time_point toc_graph_loop7 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop7 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop7 - tic_graph_loop7);
//             time_consuming.push_back(time_graph_loop7.count()*1000);
//             // std::cout << "shrink_to_fit and clear vectors: " << time_consuming[7]<< "ms" << std::endl;

//             std::chrono::steady_clock::time_point toc_graph_loop8 = std::chrono::steady_clock::now();
//             std::chrono::duration<double> time_graph_loop8 = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_loop8 - tic_graph_loop8);
//             time_consuming.push_back(time_graph_loop8.count()*1000);
//             // std::cout << "single sub graph loop consuming time: " << time_graph_loop8.count()*1000<< "ms" << std::endl;


         }
        // std::cout << "selected_idx(): " <<selected_idx.size() <<std::endl;
        // std::cout<< "selected_id:" <<selected_idx<<std::endl;
        std::cout << "Odom successfly slected by graph(percent): " << selected_idx.size()*1.0/cor_size_all * 100<<"%" <<std::endl;
    
        std::chrono::steady_clock::time_point toc_graph_correspondence = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used_init_graph_correspondence = std::chrono::duration_cast<std::chrono::duration<double>>(toc_graph_correspondence - tic_graph_correspondence);
        std::cout << "Odom's Graph     corrrespondence     consuming    time      :" << time_used_init_graph_correspondence.count() *1000<< "ms"<< std::endl;
        
        
        
        
    // }

}
//new add

void process()
{
	int now_frame =0;
	while(1)
	{
		while (!cornerLastBuf.empty() && !surfLastBuf.empty() &&
			!fullResBuf.empty() && !odometryBuf.empty())
		{
			mBuf.lock();
			while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				odometryBuf.pop();
			if (odometryBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!surfLastBuf.empty() && surfLastBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				surfLastBuf.pop();
			if (surfLastBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			while (!fullResBuf.empty() && fullResBuf.front()->header.stamp.toSec() < cornerLastBuf.front()->header.stamp.toSec())
				fullResBuf.pop();
			if (fullResBuf.empty())
			{
				mBuf.unlock();
				break;
			}

			timeLaserCloudCornerLast = cornerLastBuf.front()->header.stamp.toSec();
			timeLaserCloudSurfLast = surfLastBuf.front()->header.stamp.toSec();
			timeLaserCloudFullRes = fullResBuf.front()->header.stamp.toSec();
			timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();

			if (timeLaserCloudCornerLast != timeLaserOdometry ||
				timeLaserCloudSurfLast != timeLaserOdometry ||
				timeLaserCloudFullRes != timeLaserOdometry)
			{
				//printf("time corner %f surf %f full %f odom %f \n", timeLaserCloudCornerLast, timeLaserCloudSurfLast, timeLaserCloudFullRes, timeLaserOdometry);
				//printf("unsync messeage!");
				mBuf.unlock();
				break;
			}

			laserCloudCornerLast->clear();
			pcl::fromROSMsg(*cornerLastBuf.front(), *laserCloudCornerLast);
			cornerLastBuf.pop();

			laserCloudSurfLast->clear();
			pcl::fromROSMsg(*surfLastBuf.front(), *laserCloudSurfLast);
			surfLastBuf.pop();

			laserCloudFullRes->clear();
			pcl::fromROSMsg(*fullResBuf.front(), *laserCloudFullRes);
			fullResBuf.pop();

			q_wodom_curr.x() = odometryBuf.front()->pose.pose.orientation.x;
			q_wodom_curr.y() = odometryBuf.front()->pose.pose.orientation.y;
			q_wodom_curr.z() = odometryBuf.front()->pose.pose.orientation.z;
			q_wodom_curr.w() = odometryBuf.front()->pose.pose.orientation.w;
			t_wodom_curr.x() = odometryBuf.front()->pose.pose.position.x;
			t_wodom_curr.y() = odometryBuf.front()->pose.pose.position.y;
			t_wodom_curr.z() = odometryBuf.front()->pose.pose.position.z;
			odometryBuf.pop();

			while(!cornerLastBuf.empty())
			{
				cornerLastBuf.pop();
				//printf("drop lidar frame in mapping for real time performance \n");
			}

			mBuf.unlock();

			TicToc t_whole;

			transformAssociateToMap();

			TicToc t_shift;
			int centerCubeI = int((t_w_curr.x() + 25.0) / 50.0) + laserCloudCenWidth;
			int centerCubeJ = int((t_w_curr.y() + 25.0) / 50.0) + laserCloudCenHeight;
			int centerCubeK = int((t_w_curr.z() + 25.0) / 50.0) + laserCloudCenDepth;

			if (t_w_curr.x() + 25.0 < 0)
				centerCubeI--;
			if (t_w_curr.y() + 25.0 < 0)
				centerCubeJ--;
			if (t_w_curr.z() + 25.0 < 0)
				centerCubeK--;

			while (centerCubeI < 3)
			{
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{ 
						int i = laserCloudWidth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k]; 
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i >= 1; i--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i - 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI++;
				laserCloudCenWidth++;
			}

			while (centerCubeI >= laserCloudWidth - 3)
			{ 
				for (int j = 0; j < laserCloudHeight; j++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int i = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; i < laserCloudWidth - 1; i++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + 1 + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeI--;
				laserCloudCenWidth--;
			}

			while (centerCubeJ < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = laserCloudHeight - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j >= 1; j--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j - 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ++;
				laserCloudCenHeight++;
			}

			while (centerCubeJ >= laserCloudHeight - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int k = 0; k < laserCloudDepth; k++)
					{
						int j = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; j < laserCloudHeight - 1; j++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * (j + 1) + laserCloudWidth * laserCloudHeight * k];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeJ--;
				laserCloudCenHeight--;
			}

			while (centerCubeK < 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = laserCloudDepth - 1;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k >= 1; k--)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k - 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK++;
				laserCloudCenDepth++;
			}

			while (centerCubeK >= laserCloudDepth - 3)
			{
				for (int i = 0; i < laserCloudWidth; i++)
				{
					for (int j = 0; j < laserCloudHeight; j++)
					{
						int k = 0;
						pcl::PointCloud<PointType>::Ptr laserCloudCubeCornerPointer =
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						pcl::PointCloud<PointType>::Ptr laserCloudCubeSurfPointer =
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k];
						for (; k < laserCloudDepth - 1; k++)
						{
							laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
							laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
								laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * (k + 1)];
						}
						laserCloudCornerArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeCornerPointer;
						laserCloudSurfArray[i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k] =
							laserCloudCubeSurfPointer;
						laserCloudCubeCornerPointer->clear();
						laserCloudCubeSurfPointer->clear();
					}
				}

				centerCubeK--;
				laserCloudCenDepth--;
			}

			int laserCloudValidNum = 0;
			int laserCloudSurroundNum = 0;

			for (int i = centerCubeI - 2; i <= centerCubeI + 2; i++)
			{
				for (int j = centerCubeJ - 2; j <= centerCubeJ + 2; j++)
				{
					for (int k = centerCubeK - 1; k <= centerCubeK + 1; k++)
					{
						if (i >= 0 && i < laserCloudWidth &&
							j >= 0 && j < laserCloudHeight &&
							k >= 0 && k < laserCloudDepth)
						{ 
							laserCloudValidInd[laserCloudValidNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudValidNum++;
							laserCloudSurroundInd[laserCloudSurroundNum] = i + laserCloudWidth * j + laserCloudWidth * laserCloudHeight * k;
							laserCloudSurroundNum++;
						}
					}
				}
			}

			laserCloudCornerFromMap->clear();
			laserCloudSurfFromMap->clear();
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
				*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
			}
			int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
			int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();


			pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
			downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
			downSizeFilterCorner.filter(*laserCloudCornerStack);
			int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

			pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());
			downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
			downSizeFilterSurf.filter(*laserCloudSurfStack);
			int laserCloudSurfStackNum = laserCloudSurfStack->points.size();

			//printf("map prepare time %f ms\n", t_shift.toc());
			//printf("map corner num %d  surf num %d \n", laserCloudCornerFromMapNum, laserCloudSurfFromMapNum);
			if (laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 50)
			{
				TicToc t_opt;
				TicToc t_tree;
				kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
				kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
				//printf("build tree time %f ms \n", t_tree.toc());

				for (int iterCount = 0; iterCount < 2; iterCount++)
				{
					//new add
                    int intilize_num = (laserCloudCornerStackNum > laserCloudSurfStackNum) ? laserCloudCornerStackNum : laserCloudSurfStackNum;
                    std::vector<Eigen::Vector3d> feature_cur_point;
                    // std::cout<<"corner_cur_point_size:"<<feature_cur_point.size()<<" , corner_cur_point_capacity："<<feature_cur_point.capacity()<<std::endl;
                    std::vector<Eigen::Vector3d> feature_last_point_a;
                    std::vector<double> feature_last_point_b;
                    std::vector<double> feature_s;
                    std::vector<Corre_Match> correspondences;
                    std::vector<Vertex_Vote> selected_idx;
                    feature_cur_point.reserve(intilize_num);
                    feature_last_point_a.reserve(intilize_num);
                    feature_last_point_b.reserve(intilize_num);
                    feature_s.reserve(intilize_num);
                    correspondences.reserve(intilize_num);
                    selected_idx.reserve(intilize_num);
                    int index=0;
                    int _horizontal_scans =2117;
                    // float _ang_resolution_X = (M_PI*2) / (_horizontal_scans);;
                    // Eigen::MatrixXi graph_indexing;
                    // graph_indexing.setOnes(N_SCANS, _horizontal_scans);
                    // graph_indexing = graph_indexing * -1;
                    // float min_location;
                    // float vertical_angle_top = 2;
                    // float _ang_bottom=-24.9;
                    // float _ang_resolution_Y = M_PI / 180.0 *(vertical_angle_top - _ang_bottom) / float(N_SCANS-1);
                    // _ang_bottom = -( _ang_bottom - 0.1) * M_PI / 180.0;
                    // new add

					//ceres::LossFunction *loss_function = NULL;
					ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
					ceres::LocalParameterization *q_parameterization =
						new ceres::EigenQuaternionParameterization();
					ceres::Problem::Options problem_options;

					ceres::Problem problem(problem_options);
					problem.AddParameterBlock(parameters, 4, q_parameterization);
					problem.AddParameterBlock(parameters + 4, 3);

					TicToc t_data;
					int corner_num = 0;

					for (int i = 0; i < laserCloudCornerStackNum; i++)
					{
						pointOri = laserCloudCornerStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

						if (pointSearchSqDis[4] < 1.0)
						{ 
							std::vector<Eigen::Vector3d> nearCorners;
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
								nearCorners.push_back(tmp);
							}
							center = center / 5.0;

							Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
							for (int j = 0; j < 5; j++)
							{
								Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
								covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
							}

							Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

							// if is indeed line feature
							// note Eigen library sort eigenvalues in increasing order
							Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
							{ 
								Eigen::Vector3d point_on_line = center;
								Eigen::Vector3d point_a, point_b;
								point_a = 0.1 * unit_direction + point_on_line;
								point_b = -0.1 * unit_direction + point_on_line;

								ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								corner_num++;	
							}							
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
													laserCloudCornerFromMap->points[pointSearchInd[j]].y,
													laserCloudCornerFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}
					
					int surf_num = 0;
					for (int i = 0; i < laserCloudSurfStackNum; i++)
					{
						pointOri = laserCloudSurfStack->points[i];
						//double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
						pointAssociateToMap(&pointOri, &pointSel);
						kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
						float cx =0,cy=0,cz =0;
						Eigen::Matrix<double, 5, 3> matA0;
						Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
						if (pointSearchSqDis[4] < 1.0)
						{
							
							for (int j = 0; j < 5; j++)
							{
								matA0(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								matA0(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								matA0(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								cx += laserCloudSurfFromMap->points[pointSearchInd[j]].x;
								cy += laserCloudSurfFromMap->points[pointSearchInd[j]].y;
								cz += laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								////printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));
							}
							// find the norm of plane
							Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
							double negative_OA_dot_norm = 1 / norm.norm();
							norm.normalize();
							cx /=5;
							cy /=5;
							cz /=5;
							// Here n(pa, pb, pc) is unit norm of plane
							bool planeValid = true;
							for (int j = 0; j < 5; j++)
							{
								// if OX * n > 0.2, then plane is not fit well
								if (fabs(norm(0) * laserCloudSurfFromMap->points[pointSearchInd[j]].x +
										 norm(1) * laserCloudSurfFromMap->points[pointSearchInd[j]].y +
										 norm(2) * laserCloudSurfFromMap->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
								{
									planeValid = false;
									break;
								}
							}
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							if (planeValid)
							{
								feature_cur_point.push_back(curr_point);
								feature_last_point_a.push_back(norm);
								feature_last_point_b.push_back(negative_OA_dot_norm);
								// feature_s.push_back(s);
								// new add
								Corre_Match cor;
								cor.index = index;
								cor.src = laserCloudSurfStack->points[i];
								PointType tgt_p;
								tgt_p.x =cx;
								tgt_p.y =cy;
								tgt_p.z =cz;
								cor.tgt = tgt_p;
								cor.score = 0;
								cor.s = 0;
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
								// if(now_frame < 20)
								// {
								// 	ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
								// 	problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								
								// }
								ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(curr_point, norm, negative_OA_dot_norm);
								problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
								surf_num++;
							}
						}
						/*
						else if(pointSearchSqDis[4] < 0.01 * sqrtDis)
						{
							Eigen::Vector3d center(0, 0, 0);
							for (int j = 0; j < 5; j++)
							{
								Eigen::Vector3d tmp(laserCloudSurfFromMap->points[pointSearchInd[j]].x,
													laserCloudSurfFromMap->points[pointSearchInd[j]].y,
													laserCloudSurfFromMap->points[pointSearchInd[j]].z);
								center = center + tmp;
							}
							center = center / 5.0;	
							Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
							ceres::CostFunction *cost_function = LidarDistanceFactor::Create(curr_point, center);
							problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
						}
						*/
					}

					// if(now_frame > 20  )
                    // {
                    //     graph_based_correspondence_vote_simple(correspondences, true, selected_idx);
                    //     for(int i = 0; i < selected_idx.size(); i++)
                    //     {
                        
                    //     int idx = selected_idx[i].index;
                    //     // std::cout<<"feature_cur_point[idx]:"<<idx<<std::endl;
                    //     // ceres::CostFunction *cost_function = LidarEdgeFactor::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx], feature_s[idx]);
                    //     // problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                    //     ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(feature_cur_point[idx], feature_last_point_a[idx], feature_last_point_b[idx]);
                    //     problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);

                    //     }

                    // }

					////printf("corner num %d used corner num %d \n", laserCloudCornerStackNum, corner_num);
					////printf("surf num %d used surf num %d \n", laserCloudSurfStackNum, surf_num);

					//printf("mapping data assosiation time %f ms \n", t_data.toc());

					TicToc t_solver;
					ceres::Solver::Options options;
					options.linear_solver_type = ceres::DENSE_QR;
					options.max_num_iterations = 4;
					options.minimizer_progress_to_stdout = false;
					options.check_gradients = false;
					options.gradient_check_relative_precision = 1e-4;
					ceres::Solver::Summary summary;
					ceres::Solve(options, &problem, &summary);
					//printf("mapping solver time %f ms \n", t_solver.toc());

					////printf("time %f \n", timeLaserOdometry);
					////printf("corner factor num %d surf factor num %d\n", corner_num, surf_num);
					////printf("result q %f %f %f %f result t %f %f %f\n", parameters[3], parameters[0], parameters[1], parameters[2],
					//	   parameters[4], parameters[5], parameters[6]);
				}
				//printf("mapping optimization time %f \n", t_opt.toc());
			}
			else
			{
				ROS_WARN("time Map corner and surf num are not enough");
			}
			transformUpdate();

			TicToc t_add;
			for (int i = 0; i < laserCloudCornerStackNum; i++)
			{
				pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudCornerArray[cubeInd]->push_back(pointSel);
				}
			}

			for (int i = 0; i < laserCloudSurfStackNum; i++)
			{
				pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

				int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
				int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
				int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

				if (pointSel.x + 25.0 < 0)
					cubeI--;
				if (pointSel.y + 25.0 < 0)
					cubeJ--;
				if (pointSel.z + 25.0 < 0)
					cubeK--;

				if (cubeI >= 0 && cubeI < laserCloudWidth &&
					cubeJ >= 0 && cubeJ < laserCloudHeight &&
					cubeK >= 0 && cubeK < laserCloudDepth)
				{
					int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
					laserCloudSurfArray[cubeInd]->push_back(pointSel);
				}
			}
			//printf("add points time %f ms\n", t_add.toc());

			
			TicToc t_filter;
			for (int i = 0; i < laserCloudValidNum; i++)
			{
				int ind = laserCloudValidInd[i];

				pcl::PointCloud<PointType>::Ptr tmpCorner(new pcl::PointCloud<PointType>());
				downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
				downSizeFilterCorner.filter(*tmpCorner);
				laserCloudCornerArray[ind] = tmpCorner;

				pcl::PointCloud<PointType>::Ptr tmpSurf(new pcl::PointCloud<PointType>());
				downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
				downSizeFilterSurf.filter(*tmpSurf);
				laserCloudSurfArray[ind] = tmpSurf;
			}
			//printf("filter time %f ms \n", t_filter.toc());
			
			TicToc t_pub;
			//publish surround map for every 5 frame
			if (frameCount % 5 == 0)
			{
				laserCloudSurround->clear();
				for (int i = 0; i < laserCloudSurroundNum; i++)
				{
					int ind = laserCloudSurroundInd[i];
					*laserCloudSurround += *laserCloudCornerArray[ind];
					*laserCloudSurround += *laserCloudSurfArray[ind];
				}

				sensor_msgs::PointCloud2 laserCloudSurround3;
				pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
				laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudSurround3.header.frame_id = "/camera_init";
				pubLaserCloudSurround.publish(laserCloudSurround3);
			}

			if (frameCount % 20 == 0)
			{
				pcl::PointCloud<PointType> laserCloudMap;
				for (int i = 0; i < 4851; i++)
				{
					laserCloudMap += *laserCloudCornerArray[i];
					laserCloudMap += *laserCloudSurfArray[i];
				}
				sensor_msgs::PointCloud2 laserCloudMsg;
				pcl::toROSMsg(laserCloudMap, laserCloudMsg);
				laserCloudMsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudMsg.header.frame_id = "/camera_init";
				pubLaserCloudMap.publish(laserCloudMsg);
			}

			int laserCloudFullResNum = laserCloudFullRes->points.size();
			for (int i = 0; i < laserCloudFullResNum; i++)
			{
				pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
			}

			sensor_msgs::PointCloud2 laserCloudFullRes3;
			pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
			laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			laserCloudFullRes3.header.frame_id = "/camera_init";
			pubLaserCloudFullRes.publish(laserCloudFullRes3);

			//printf("mapping pub time %f ms \n", t_pub.toc());

			printf("whole mapping time %f ms +++++\n", t_whole.toc());

			// // correct the rotation difference
			// //roll
			// double sinr_cosp = 2 * (q_w_curr.w() * q_w_curr.x() + q_w_curr.y()*q_w_curr.z());
			// double cosr_cosp = 1 - 2 * (q_w_curr.x() * q_w_curr.x() + q_w_curr.y() * q_w_curr.y());
    		// double roll = std::atan2(sinr_cosp, cosr_cosp);
 
    		// // pitch (y-axis rotation)
    		// double sinp = 2 * (q_w_curr.w() * q_w_curr.y() - q_w_curr.z() * q_w_curr.x());
    		// double pitch;
			// if (std::abs(sinp) >= 1)
        	// 	pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    		// else
        	// 	pitch = std::asin(sinp);
 
			// // yaw (z-axis rotation)
			// double siny_cosp = 2 * (q_w_curr.w() * q_w_curr.z() + q_w_curr.x() * q_w_curr.y());
			// double cosy_cosp = 1 - 2 * (q_w_curr.y() * q_w_curr.y() + q_w_curr.z() * q_w_curr.z());
			// double yaw = std::atan2(siny_cosp, cosy_cosp);
			// // double w_new = q_w_curr.w();
			// // double x_new = q_w_curr.x();
			// // double y_new = q_w_curr.y();
			// // double z_new = q_w_curr.z();

			// roll = roll + M_PI / 2;
			// yaw = yaw + M_PI / 2;


			// double cy = cos(yaw * 0.5);
			// double sy = sin(yaw * 0.5);
			// double cp = cos(pitch * 0.5);
			// double sp = sin(pitch * 0.5);
			// double cr = cos(roll * 0.5);
			// double sr = sin(roll * 0.5);
		
			// Eigen::Quaterniond q_after;
			// q_after.w() = cy * cp * cr + sy * sp * sr;
			// q_after.x() = cy * cp * sr - sy * sp * cr;
			// q_after.y() = sy * cp * sr + cy * sp * cr;
			// q_after.z() = sy * cp * cr - cy * sp * sr;


			nav_msgs::Odometry odomAftMapped;
			odomAftMapped.header.frame_id = "/camera_init";
			odomAftMapped.child_frame_id = "/aft_mapped";
			odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
			odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
			odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
			odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
			odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
			// odomAftMapped.pose.pose.orientation.x = q_w_curr.y();
			// odomAftMapped.pose.pose.orientation.y = -q_w_curr.x();
			// odomAftMapped.pose.pose.orientation.z = q_w_curr.w();
			// odomAftMapped.pose.pose.orientation.w = -q_w_curr.z();

			// odomAftMapped.pose.pose.orientation.x = q_after.y();
			// odomAftMapped.pose.pose.orientation.y = -q_after.x();
			// odomAftMapped.pose.pose.orientation.z = q_after.w();
			// odomAftMapped.pose.pose.orientation.w = -q_after.z();
			odomAftMapped.pose.pose.position.x = t_w_curr.x();
			odomAftMapped.pose.pose.position.y = t_w_curr.y();
			odomAftMapped.pose.pose.position.z = t_w_curr.z();
			pubOdomAftMapped.publish(odomAftMapped);

			//--------------write odometry file to evaluate------------
			Eigen::Quaterniond q_from_odom;
			q_from_odom.w() = odomAftMapped.pose.pose.orientation.w;
			q_from_odom.x() = odomAftMapped.pose.pose.orientation.x;
			q_from_odom.y() = odomAftMapped.pose.pose.orientation.y;
			q_from_odom.z() = odomAftMapped.pose.pose.orientation.z;
			Eigen::Matrix3d R_map = q_from_odom.toRotationMatrix();
			if(init_flag)
			{
				H_init << R_map.row(0)[0], R_map.row(0)[1], R_map.row(0)[2], odomAftMapped.pose.pose.position.x,
						  R_map.row(1)[0], R_map.row(1)[1], R_map.row(1)[2], odomAftMapped.pose.pose.position.y,
						  R_map.row(2)[0], R_map.row(2)[1], R_map.row(2)[2], odomAftMapped.pose.pose.position.z,
						  0,0,0,1;
				init_flag = false;
			}

			H << R_map.row(0)[0], R_map.row(0)[1], R_map.row(0)[2], odomAftMapped.pose.pose.position.x,
				 R_map.row(1)[0], R_map.row(1)[1], R_map.row(1)[2], odomAftMapped.pose.pose.position.y,
				 R_map.row(2)[0], R_map.row(2)[1], R_map.row(2)[2], odomAftMapped.pose.pose.position.z,
				 0,0,0,1;
			H = H_init.inverse() * H;

			std::ofstream foutC(RESULT_PATH, std::ios::app);
			foutC.setf(std::ios::scientific, std::ios::floatfield);
			foutC.precision(6);
			
			for (int row=0; row<3; row++)
			{
				
				for(int col=0; col<4; col++)
				{
					if(row==2 && col==3)
					{
						foutC << H.row(row)[col] <<std::endl;
					}
					else
					{
						foutC << H.row(row)[col] << " ";
					}
				}
			}
			foutC.close();
			//--------------write odometry file to evaluate------------
			geometry_msgs::PoseStamped laserAfterMappedPose;
			laserAfterMappedPose.header = odomAftMapped.header;
			laserAfterMappedPose.pose = odomAftMapped.pose.pose;
			laserAfterMappedPath.header.stamp = odomAftMapped.header.stamp;
			laserAfterMappedPath.header.frame_id = "/camera_init";
			laserAfterMappedPath.poses.push_back(laserAfterMappedPose);
			pubLaserAfterMappedPath.publish(laserAfterMappedPath);
			
			static tf::TransformBroadcaster br;
			tf::Transform transform;
			tf::Quaternion q;
			transform.setOrigin(tf::Vector3(t_w_curr(0),
											t_w_curr(1),
											t_w_curr(2)));
			q.setW(q_w_curr.w());
			q.setX(q_w_curr.x());
			q.setY(q_w_curr.y());
			q.setZ(q_w_curr.z());
			transform.setRotation(q);
			br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

			frameCount++;
			now_frame++;
		}
		std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
	}
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserMapping");
	ros::NodeHandle nh;

	float lineRes = 0;
	float planeRes = 0;
	nh.param<float>("mapping_line_resolution", lineRes, 0.4);
	nh.param<float>("mapping_plane_resolution", planeRes, 0.8);
	nh.param<std::string>("RESULT_PATH", RESULT_PATH, " ");
	std::cout<<"RESULT_PATH:" <<RESULT_PATH <<std::endl;
	//printf("line resolution %f plane resolution %f \n", lineRes, planeRes);
	downSizeFilterCorner.setLeafSize(lineRes, lineRes,lineRes);
	downSizeFilterSurf.setLeafSize(planeRes, planeRes, planeRes);

	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, laserCloudCornerLastHandler);

	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, laserCloudSurfLastHandler);

	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 100, laserOdometryHandler);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100, laserCloudFullResHandler);

	pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100);

	pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 100);

	pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 100);

	pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 100);

	pubOdomAftMappedHighFrec = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 100);

	pubLaserAfterMappedPath = nh.advertise<nav_msgs::Path>("/aft_mapped_path", 100);

	for (int i = 0; i < laserCloudNum; i++)
	{
		laserCloudCornerArray[i].reset(new pcl::PointCloud<PointType>());
		laserCloudSurfArray[i].reset(new pcl::PointCloud<PointType>());
	}

	std::thread mapping_process{process};

	ros::spin();

	return 0;
}
