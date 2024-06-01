#pragma once

#include <cmath>

#include <pcl/point_types.h>

typedef pcl::PointXYZI PointType;

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}

//new add
typedef struct {
    int index;
    // int tgt_idx;
    pcl::PointXYZI src;
    pcl::PointXYZI tgt;
    // Eigen::Vector3f src_norm;
    // Eigen::Vector3f tgt_norm;
    // Eigen::Matrix3f covariance_src;
    // Eigen::Matrix3f covariance_tgt;
    float score;
    float s; // optimization weight
}Corre_Match;

typedef struct 
{
    int vertex_index;
    std::vector<int> connected_vertex_index;
    int degree;
}Vertex_Attribute;

typedef struct{
    int index;
    float score;
}Vertex_Vote;

// bool compare_degree(const Vertex_Attribute ob1, const Vertex_Attribute ob2)
// {
//     return ob1.degree > ob2.degree;
// }

struct compare_score{
  bool operator()(Vertex_Vote const &ob1, Vertex_Vote const &ob2){return ob1.score > ob2.score;}
};
//new add
