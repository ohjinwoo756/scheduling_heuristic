#include "gurobi_c++.h"
#include <vector>
#include <string>
#include <iostream>
#include <fcntl.h>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "DS.pb.h"
#include "caffe.pb.h"

float const_comm_init = 0;
float const_comm_delay = 1;

float map_alpha = 0.0053;
float map_beta = 569.25;
float unmap_alpha = 0.0024;
float unmap_beta = 526.39;
float memcpy_alpha = 6 * pow(10,-10);
float memcpy_beta = 7 * pow(10,-5);
float memcpy_gamma = 1.6979;

struct result {
    std::string layer_name;
    std::string mapped;
    float start_time;
    float end_time;
    float comm_time;
};

// for porting to Python
int num_layer;

// Global result for Gantt chart output
std::vector<struct result> res; 
struct result data;

// data vector for merging
int* data_for_merging;
float throughput;

// For converting from string to char*
char* ptr;
std::vector<char> global_writable;

std::vector<result> return_result   (std::vector<result> _result_set)
{
    return _result_set;
}

// Body
extern "C"
{
    int ilp_util(char* model_file, char* estimation_file, int CPU_utilization, int GPU_utilization)  {
        int mode = 1;
        int num_of_PE = 0;
        int num_of_CPU = 0;
        DS::network * time = new DS::network;
        caffe::NetParameter * caffe_model = new caffe::NetParameter;
        
        // PE profile time
        std::vector<float> CPU_time;
        std::vector<float> GPU_time;
        std::vector<float> dsp_time;
        std::vector<float> npu_time;
        std::vector<float> CPU2_time;
        std::vector<float> CPU3_time;
        std::vector<float> CPU4_time;
        std::vector<std::vector<float>> layer_comm;
        
        std::vector<GRBVar> layer_CPU;
        std::vector<GRBVar> layer_CPU2;
        std::vector<GRBVar> layer_CPU3;
        std::vector<GRBVar> layer_CPU4;
        std::vector<GRBVar> layer_GPU;
        std::vector<GRBVar> layer_npu;
        std::vector<GRBVar> layer_dsp;

        // new CPU cluster
        std::vector<GRBVar> post_CPU;
        std::vector<GRBVar> not_post_CPU;
        std::vector<GRBVar> CPU_cluster;
        // new CPU2 cluster
        std::vector<GRBVar> post_CPU2;
        std::vector<GRBVar> not_post_CPU2;
        std::vector<GRBVar> CPU2_cluster;
        // new CPU3 cluster
        std::vector<GRBVar> post_CPU3;
        std::vector<GRBVar> not_post_CPU3;
        std::vector<GRBVar> CPU3_cluster;
        // new CPU4 cluster
        std::vector<GRBVar> post_CPU4;
        std::vector<GRBVar> not_post_CPU4;
        std::vector<GRBVar> CPU4_cluster;
        // new GPU cluster
        std::vector<GRBVar> post_GPU;
        std::vector<GRBVar> not_post_GPU;
        std::vector<GRBVar> GPU_cluster;
        
		// Future works ------------
        // new npu cluster
        std::vector<GRBVar> post_npu;
        std::vector<GRBVar> not_post_npu;
        std::vector<GRBVar> NPU_cluster;
        // new dsp cluster
        std::vector<GRBVar> post_dsp;
        std::vector<GRBVar> not_post_dsp;
        std::vector<GRBVar> DSP_cluster;

        // time variables for ilp
        std::vector<std::string> layer;
        std::vector<std::vector<std::string>> bottom_layer;
        std::vector<GRBVar> layer_start;
        std::vector<GRBVar> layer_end;
        
        // mapped at same PE
        std::vector<std::vector<std::vector<GRBVar>>> same;
        // [ith layer][jth layer][0] -> ith layer && jth layer is both mapped at same PE, binary
        // [ith layer][jth layer][1] -> ith layer && jth layer is both mapped at CPU, binary
        // [ith layer][jth layer][2] -> ith layer && jth layer is both mapped at GPU, binary
        // [ith layer][jth layer][3] -> ith layer && jth layer is both mapped at CPU2, binary
        // [ith layer][jth layer][4] -> ith layer && jth layer is both mapped at CPU3, binary
        // [ith layer][jth layer][5] -> ith layer && jth layer is both mapped at CPU4, binary
        
        // map & unmap variables
        std::vector<std::vector<std::vector<GRBVar>>> map;
        std::vector<std::vector<std::vector<GRBVar>>> unmap;
        // [ith layer][jth layer][0] -> from CPU1 unmap
        // [ith layer][jth layer][1] -> from CPU1 unmap
        // [ith layer][jth layer][2] -> from CPU1 unmap
        // [ith layer][jth layer][3] -> from CPU1 unmap
        std::vector<std::vector<std::vector<GRBVar>>> memcpy;
        // [ith layer][jth layer][0] -> from CPU1 memcpy
        // [ith layer][jth layer][1] -> from CPU1 memcpy
        // [ith layer][jth layer][2] -> from CPU1 memcpy
        // [ith layer][jth layer][3] -> from CPU1 memcpy
        // transition variable for minimizing needless layer moving
        std::vector<std::vector<GRBVar>> PE_changed;
        
        //vectors for result
        std::vector<std::pair<std::string,std::vector<int>>> mem_size;
        std::vector<std::vector<result>> result_set;
        
        int num_of_CPU_core = 4;
        int num_of_CPU2_core = 0;
        int num_of_CPU3_core = 0;
        int num_of_CPU4_core = 0;
        
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        
        int model_descriptor = open(model_file, O_RDONLY);
        
        google::protobuf::io::FileInputStream model_input(model_descriptor);
        model_input.SetCloseOnDelete( true );
        
        if (!google::protobuf::TextFormat::Parse(&model_input, caffe_model))    {
            // std::cerr << "Failed to parse model file!" << std::endl;
            return -1;
        }
        
        int time_descriptor = open(estimation_file, O_RDONLY);
        google::protobuf::io::FileInputStream time_input(time_descriptor);
        time_input.SetCloseOnDelete( true );
        
        if (!google::protobuf::TextFormat::Parse(&time_input, time))    {
            // std::cerr << "Failed to parse time file!" << std::endl;
            return -1;
        }
        
        //check PE num
        if (time->layer(0).has_cpu())    {
            num_of_PE++;
            num_of_CPU++;
        }
        if (time->layer(0).has_gpu())   {
            num_of_PE++;
        }
        if (time->layer(0).has_dsp())    {
            num_of_PE++;
        }
        if (time->layer(0).has_npu())   {
            num_of_PE++;
        }
        if (time->layer(0).has_cpu2())    {
            num_of_PE++;
            num_of_CPU++;
            
            if(time->layer(1).cpu() != time->layer(1).cpu2() )  {
                num_of_CPU_core = 3;
                num_of_CPU2_core = 1;
            }
            else    {
                num_of_CPU_core = 2;
                num_of_CPU2_core = 2;
                std::cerr << " test 22" << std::endl;
            }
        }
        if (time->layer(0).has_cpu3())    {
            num_of_PE++;
            num_of_CPU++;
            
            num_of_CPU_core = 2;
            num_of_CPU2_core = 1;
            num_of_CPU3_core = 1;
        }
        if (time->layer(0).has_cpu4())    {
            num_of_PE++;
            num_of_CPU++;
            
            num_of_CPU_core = 1;
            num_of_CPU2_core = 1;
            num_of_CPU3_core = 1;
            num_of_CPU4_core = 1;
        }
        
        int temp_i = 0;
        // set layers based on time estimations.
        for (int i = 0; i < caffe_model->layer_size(); i ++)  {
            if (caffe_model->layer(i).name() == time->layer(temp_i).name()) {
                layer.push_back(caffe_model->layer(i).name());
                // set pe time here
                if (time->layer(temp_i).has_cpu())    {
                    CPU_time.push_back(time->layer(temp_i).cpu());
                }
                if (time->layer(temp_i).has_gpu())   {
                    GPU_time.push_back(time->layer(temp_i).gpu());
                }
                if (time->layer(temp_i).has_dsp())    {
                    dsp_time.push_back(time->layer(temp_i).dsp());
                }
                if (time->layer(temp_i).has_npu())   {
                    npu_time.push_back(time->layer(temp_i).npu());
                }
                if (time->layer(temp_i).has_cpu2())    {
                    CPU2_time.push_back(time->layer(temp_i).cpu2());
                }
                if (time->layer(temp_i).has_cpu3())    {
                    CPU3_time.push_back(time->layer(temp_i).cpu3());
                }
                if (time->layer(temp_i).has_cpu4())    {
                    CPU4_time.push_back(time->layer(temp_i).cpu4());
                }
                temp_i++;
                
                std::vector<std::string> temp_bottom;
                std::vector<std::string> temp_bottom_type;
                if (caffe_model->layer(i).bottom_size())    {
                    for (int k = 0; k < caffe_model->layer(i).bottom_size(); k ++)   {
                        temp_bottom.push_back(caffe_model->layer(i).bottom(k));
                    }
                }
                else    {
                    temp_bottom.push_back("start");
                }
                
                for (int j = 0; j < i; j++)   {
                    for (int k = 0; k < temp_bottom.size(); k ++)    {
                        if  (temp_bottom[k] == caffe_model->layer(j).name())   {
                            temp_bottom_type.push_back(caffe_model->layer(j).type());
                        }
                    }
                }
                
                int while_switch = 1;
                if (temp_bottom_type.size() < 1)    {
                    while_switch = 0;
                }
                while (while_switch)   {
                    for (int j = 0; j < std::min(2,(int)temp_bottom_type.size()); j ++)   {
                        if (temp_bottom_type[temp_bottom_type.size() -1 -j] == "Concat")    {
                            std::string temp_layer_name = temp_bottom[temp_bottom.size() -1 -j];
                            temp_bottom.pop_back();
                            for (int k = 0; k < i; k ++)   {
                                if  (temp_layer_name == caffe_model->layer(k).name())   {
                                    for (int l = 0; l < caffe_model->layer(k).bottom_size(); l ++)   {
                                        temp_bottom.push_back(caffe_model->layer(k).bottom(l));
                                    }
                                }
                            }
                        }
                    }
                    temp_bottom_type.clear();
                    for (int j = 0; j < temp_bottom.size(); j++)   {
                        for (int k = 0; k < i; k ++)    {
                            if  (temp_bottom[j] == caffe_model->layer(k).name())   {
                                temp_bottom_type.push_back(caffe_model->layer(k).type());
                            }
                        }
                    }
                    if (temp_bottom.size() <= 1)   {
                        while_switch = 0;
                    }
                    else    {
                        if (temp_bottom_type[temp_bottom_type.size()-1] != "Concat" && temp_bottom_type[temp_bottom_type.size()-2] != "Concat")   {
                            while_switch = 0;
                        }
                    }
                } // while

                bottom_layer.push_back(temp_bottom);

            } // if
        } // for
        
        // check whether all the time layers are set to layers.
        if (time->layer_size() != layer.size()){
//            std::cerr << "Model layers are not matching with time layers!  Check the naming of layers first!" << std::endl;
            return -2;
        }
        
        //set all the initial previous_contidions 0
        for (int i = 0; i < layer.size(); i++){
            std::vector<float> temp_float_vector;
            temp_float_vector.reserve(layer.size());
            
            for (int j = 0; j < layer.size(); j++){
                temp_float_vector.push_back(0.0);
            }
            layer_comm.push_back(temp_float_vector);
        }
        
        //set initial data size
        if (caffe_model->layer(0).has_input_param())    {
            int temp1 = caffe_model->layer(0).input_param().shape(0).dim(1);
            int temp2 = caffe_model->layer(0).input_param().shape(0).dim(2);
            int temp3 = caffe_model->layer(0).input_param().shape(0).dim(3);
            std::vector<int> size = {temp1,temp2,temp3};
            mem_size.push_back(std::make_pair(caffe_model->layer(0).name(),size));
        }
        else if (caffe_model->layer(0).has_transform_param())   {
            int temp2 = caffe_model->layer(0).transform_param().crop_size();
            std::vector<int> size = {3,temp2,temp2};
            mem_size.push_back(std::make_pair(caffe_model->layer(0).name(),size));
        }
        
        //(width_or_height + 2 * pad - kernel_size) / stride + 1
        for (int i = 1; i < caffe_model->layer_size(); i ++) {
            unsigned int temp_output = 0;
            unsigned int temp_width = 0;
            unsigned int temp_height = 0;
            
            for(int k = 0; k < caffe_model->layer(i).bottom_size(); k++)   {
                
                if (mem_size[i-k-1].second[0] > temp_output)  {
                    temp_output = mem_size[i-1].second[0];
                }
                if (mem_size[i-k-1].second[1] > temp_width)  {
                    temp_width = mem_size[i-1].second[1];
                }
                if (mem_size[i-k-1].second[2] > temp_height)  {
                    temp_height = mem_size[i-1].second[2];
                }
            }
            
            //conv_param
            if (caffe_model->layer(i).convolution_param().has_num_output())    {
                temp_output = caffe_model->layer(i).convolution_param().num_output();
            }
            if (caffe_model->layer(i).convolution_param().kernel_size_size())    {
                temp_width -= caffe_model->layer(i).convolution_param().kernel_size(0);
                temp_height -= caffe_model->layer(i).convolution_param().kernel_size(0);
            }
            if (caffe_model->layer(i).convolution_param().pad_size())    {
                temp_width += (2 * caffe_model->layer(i).convolution_param().pad(0));
                temp_height += (2 * caffe_model->layer(i).convolution_param().pad(0));
            }
            if (caffe_model->layer(i).convolution_param().stride_size())    {
                temp_width /= caffe_model->layer(i).convolution_param().stride(0);
                temp_height /= caffe_model->layer(i).convolution_param().stride(0);
            }
            if (caffe_model->layer(i).convolution_param().kernel_size_size())    {
                temp_width += 1;
                temp_height += 1;
            }
            //pooling_param
            if (caffe_model->layer(i).has_pooling_param())    {
                for (int j = 1; j < layer.size(); j ++)    {
                    if (caffe_model->layer(i).name() == layer[j])  {
                        temp_output *= bottom_layer[j].size();
                    }
                }
                temp_width -= caffe_model->layer(i).pooling_param().kernel_size();
                temp_height -= caffe_model->layer(i).pooling_param().kernel_size();
                
                temp_width += (2 * caffe_model->layer(i). pooling_param().pad());
                temp_height += (2 * caffe_model->layer(i).pooling_param().pad());
                
                temp_width /= caffe_model->layer(i).pooling_param().stride();
                temp_height /= caffe_model->layer(i).pooling_param().stride();
                temp_width += 1;
                temp_height += 1;
            }
            
            //inner_product_param
            if (caffe_model->layer(i).has_inner_product_param())    {
                temp_output = caffe_model->layer(i).inner_product_param().num_output();
                temp_width = 1;
                temp_height = 1;
            }
            std::vector<int> size = {(int)temp_output,(int)temp_width,(int)temp_height};
//            std::cerr <<"layer : "<<caffe_model->layer(i).name() <<"out : " <<temp_output <<"width : " << temp_width << "height : " <<temp_height<<std::endl;
            mem_size.push_back(std::make_pair(caffe_model->layer(i).name(),size));
        }
        for (int i = 0; i < bottom_layer.size(); i ++)  {
            for (int j = 0; j < layer.size(); j ++)  {
                for (int l = 0; l < bottom_layer[i].size(); l ++)  {
                    if (bottom_layer[i][l] == layer[j]) {
                        for (int k = 0; k < mem_size.size(); k ++)  {
                            if (mem_size[k].first == layer[j]) {
//                                std:: cout <<"mem_size layer : " << mem_size[k].first << std::endl;
                                float temp_commtime = 0;
                                temp_commtime = (mem_size[k].second[0] * mem_size[k].second[1] * mem_size[k].second[2]);
//                                std:: cout << i<<"th bottomlayer is " <<layer[j] << " to " << layer[i] << "mem_size : " << temp_commtime << std::endl;
                                layer_comm[j][i] = temp_commtime;
                            }
                        }
//                        std:: cerr << "map : " <<(map_alpha * layer_comm[j][i] + map_beta) << '\t';
//                        std:: cerr << "unmap : " <<(unmap_alpha * layer_comm[j][i] + unmap_beta) << std::endl;
                    }
                }
            }
        }
        
        for (int i = 0; i < layer.size(); i++){
            std::vector<GRBVar> temp_same_set;
            temp_same_set.reserve(num_of_PE+1);
            
            std::vector<GRBVar> temp_map_set;
            temp_map_set.reserve(1);
            
            std::vector<GRBVar> temp_unmap_set;
            temp_unmap_set.reserve(num_of_CPU);
            
            std::vector<GRBVar> temp_memcpy_set;
            temp_memcpy_set.reserve(num_of_CPU);
            
            std::vector<GRBVar> temp_PE_changed;
            temp_PE_changed.reserve(num_of_PE);
            
            std::vector<std::vector<GRBVar>> temp_GRB_vector;
            temp_GRB_vector.reserve(layer.size());
            
            std::vector<std::vector<GRBVar>> temp_map_vector;
            temp_map_vector.reserve(layer.size());
            
            std::vector<std::vector<GRBVar>> temp_unmap_vector;
            temp_unmap_vector.reserve(layer.size());
            
            std::vector<std::vector<GRBVar>> temp_memcpy_vector;
            temp_memcpy_vector.reserve(layer.size());
            for (int j = 0; j < layer.size(); j++){
                temp_GRB_vector.push_back(temp_same_set);
                temp_map_vector.push_back(temp_map_set);
                temp_unmap_vector.push_back(temp_unmap_set);
                temp_memcpy_vector.push_back(temp_memcpy_set);
            }
            same.push_back(temp_GRB_vector);
            map.push_back(temp_map_vector);
            unmap.push_back(temp_unmap_vector);
            memcpy.push_back(temp_memcpy_vector);
            PE_changed.push_back(temp_PE_changed);
        }
       
        //gurobi start
        try {
            GRBEnv env = GRBEnv();
            GRBModel model = GRBModel(env);
            
            // Create variables
            //CPU mapped
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = temp_string + "_CPU";
                GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                layer_CPU.push_back(CPU);
            }
            //post_CPU
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = "post_layer_of_" + temp_string + "_CPU";
                GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                post_CPU.push_back(CPU);
            }
            //not_post_CPU
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = "post_" + temp_string + "_not_CPU";
                GRBVar not_CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                not_post_CPU.push_back(not_CPU);
            }
            //CPU_cluster
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = temp_string + "_CPUclust";
                GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                CPU_cluster.push_back(clust);
            }
            
            //GPU mapped
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = temp_string + "_GPU";
                GRBVar GPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                layer_GPU.push_back(GPU);
            }
            //post_GPU
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = "post_layer_of_" + temp_string + "_GPU";
                GRBVar GPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                post_GPU.push_back(GPU);
            }
            //not_post_GPU
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = "post_" + temp_string + "_not_GPU";
                GRBVar not_GPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                not_post_GPU.push_back(not_GPU);
            }
            //GPU_cluster
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = temp_string + "_GPUclust";
                GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                GPU_cluster.push_back(clust);
            }
            
            if (CPU2_time.size())    {
                //CPU2 mapped
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_CPU2";
                    GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    layer_CPU2.push_back(CPU);
                }
                //post_CPU2
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_layer_of_" + temp_string + "_CPU2";
                    GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    post_CPU2.push_back(CPU);
                }
                //not_post_CPU2
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_" + temp_string + "_not_CPU2";
                    GRBVar not_CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    not_post_CPU2.push_back(not_CPU);
                }
                //CPU2_cluster
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_CPU2clust";
                    GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    CPU2_cluster.push_back(clust);
                }
            }
            if (CPU3_time.size())    {
                //CPU3 mapped
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_CPU3";
                    GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    layer_CPU3.push_back(CPU);
                }
                //post_CPU3
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_layer_of_" + temp_string + "_CPU3";
                    GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    post_CPU3.push_back(CPU);
                }
                //not_post_CPU3
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_" + temp_string + "_not_CPU3";
                    GRBVar not_CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    not_post_CPU3.push_back(not_CPU);
                }
                //CPU3_cluster
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_CPU3clust";
                    GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    CPU3_cluster.push_back(clust);
                }
            }
            if (CPU4_time.size())    {
                //CPU4 mapped
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_CPU4";
                    GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    layer_CPU4.push_back(CPU);
                }
                //post_CPU4
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_layer_of_" + temp_string + "_CPU4";
                    GRBVar CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    post_CPU4.push_back(CPU);
                }
                //not_post_CPU4
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_" + temp_string + "_not_CPU4";
                    GRBVar not_CPU = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    not_post_CPU4.push_back(not_CPU);
                }
                //CPU4_cluster
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_CPU4clust";
                    GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    CPU4_cluster.push_back(clust);
                }
            }
            if (npu_time.size())    {
                //NPU mapped
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_npu";
                    GRBVar npu = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    layer_npu.push_back(npu);
                }
                //post_npu
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_layer_of_" + temp_string + "_npu";
                    GRBVar npu = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    post_npu.push_back(npu);
                }
                //not_post_npu
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_" + temp_string + "_not_npu";
                    GRBVar not_npu = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    not_post_npu.push_back(not_npu);
                }
                //NPU_cluster
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_NPUclust";
                    GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    NPU_cluster.push_back(clust);
                }
            }
            
            if (dsp_time.size())    {
                //GPU mapped
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_dsp";
                    GRBVar dsp = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    layer_dsp.push_back(dsp);
                }
                //post_dsp
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_layer_of_" + temp_string + "_dsp";
                    GRBVar dsp = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    post_dsp.push_back(dsp);
                }
                //not_post_dsp
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = "post_" + temp_string + "_not_dsp";
                    GRBVar not_dsp = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    not_post_dsp.push_back(not_dsp);
                }
                //DSP_cluster
                for (int i = 0; i < layer.size(); i ++){
                    std::string temp_string = layer[i];
                    temp_string = temp_string + "_DSPclust";
                    GRBVar clust = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, temp_string);
                    DSP_cluster.push_back(clust);
                }
            }
            
            //start_time
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = temp_string + "_start";
                GRBVar start_time = model.addVar(0.0, 999999999.0, 0.0, GRB_CONTINUOUS, temp_string);
                layer_start.push_back(start_time);
            }
            //end_time
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i];
                temp_string = temp_string + "_end";
                GRBVar end_time = model.addVar(0.0, 999999999.0, 0.0, GRB_CONTINUOUS, temp_string);
                layer_end.push_back(end_time);
            }
            
            //continue
            GRBVar CPU_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "CPU end-start is linked");
            GRBVar GPU_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "GPU end-start is linked");
            GRBVar CPU2_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "CPU2 end-start is linked");
            GRBVar CPU3_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "CPU3 end-start is linked");
            GRBVar CPU4_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "CPU4 end-start is linked");
            GRBVar NPU_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "NPU end-start is linked");
            GRBVar DSP_continue = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "DSP end-start is linked");
            
            //mapped at same PE
            for (int i = 0; i < layer.size(); i ++){
                for (int j = 0; j < layer.size(); j ++){
                    std::string temp_string1 = layer[i];
                    std::string temp_string2 = layer[j];
                    std::string temp_string = temp_string1 + "_and_" + temp_string2 + "_are_same_PE";
                    GRBVar ijsame = model.addVar(0.0, 1.0 ,0.0, GRB_BINARY, temp_string);
                    same[i][j].push_back(ijsame);
                    temp_string = temp_string1 + "_and_" + temp_string2 + "_are_CPU";
                    GRBVar ijCPU = model.addVar(0.0, 1.0 ,0.0, GRB_BINARY, temp_string);
                    same[i][j].push_back(ijCPU);
                    temp_string = temp_string1 + "_and_" + temp_string2 + "_are_GPU";
                    GRBVar ijGPU = model.addVar(0.0, 1.0 ,0.0, GRB_BINARY, temp_string);
                    same[i][j].push_back(ijGPU);
                    temp_string = temp_string1 + "_and_" + temp_string2 + "_are_CPU2";
                    GRBVar ijCPU2 = model.addVar(0.0, 1.0 ,0.0, GRB_BINARY, temp_string);
                    same[i][j].push_back(ijCPU2);
                    temp_string = temp_string1 + "_and_" + temp_string2 + "_are_CPU3";
                    GRBVar ijCPU3 = model.addVar(0.0, 1.0 ,0.0, GRB_BINARY, temp_string);
                    same[i][j].push_back(ijCPU3);
                    temp_string = temp_string1 + "_and_" + temp_string2 + "_are_CPU4";
                    GRBVar ijCPU4 = model.addVar(0.0, 1.0 ,0.0, GRB_BINARY, temp_string);
                    same[i][j].push_back(ijCPU4);
                }
            }
            
            //map
            for (int i = 0; i < layer.size(); i ++){
                for (int j = 0; j < layer.size(); j ++){
                    std::string temp_string1 = layer[i];
                    std::string temp_string2 = layer[j];
                    std::string temp_string = "GPU" + temp_string1 + " -> " + temp_string2 + "map";
                    GRBVar map1 = model.addVar(0.0, 1.0 ,0.0, GRB_INTEGER, temp_string);
                    map[i][j].push_back(map1);
                }
            }
            
            //unmap
            for (int i = 0; i < layer.size(); i ++){
                for (int j = 0; j < layer.size(); j ++){
                    std::string temp_string1 = layer[i];
                    std::string temp_string2 = layer[j];
                    std::string temp_string ="CPU1 " + temp_string1 + " -> " + temp_string2 + " unmap";
                    GRBVar unmap1 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    unmap[i][j].push_back(unmap1);
                    temp_string ="CPU2 " + temp_string1 + " -> " + temp_string2 + " unmap";
                    GRBVar unmap2 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    unmap[i][j].push_back(unmap2);
                    temp_string ="CPU3 " + temp_string1 + " -> " + temp_string2 + " unmap";
                    GRBVar unmap3 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    unmap[i][j].push_back(unmap3);
                    temp_string ="CPU4 " + temp_string1 + " -> " + temp_string2 + " unmap";
                    GRBVar unmap4 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    unmap[i][j].push_back(unmap4);
                }
            }
            
            //CPU to CPU memcpy
            for (int i = 0; i < layer.size(); i ++){
                for (int j = 0; j < layer.size(); j ++){
                    std::string temp_string1 = layer[i];
                    std::string temp_string2 = layer[j];
                    std::string temp_string ="CPU1 " + temp_string1 + " -> " + temp_string2 + " memcpy";
                    GRBVar memcpy1 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    memcpy[i][j].push_back(memcpy1);
                    temp_string ="CPU2 " + temp_string1 + " -> " + temp_string2 + " memcpy";
                    GRBVar memcpy2 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    memcpy[i][j].push_back(memcpy2);
                    temp_string ="CPU3 " + temp_string1 + " -> " + temp_string2 + " memcpy";
                    GRBVar memcpy3 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    memcpy[i][j].push_back(memcpy3);
                    temp_string ="CPU4 " + temp_string1 + " -> " + temp_string2 + " memcpy";
                    GRBVar memcpy4 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                    memcpy[i][j].push_back(memcpy4);
                }
            }
            
            //PE_changed
            for (int i = 0; i < layer.size(); i ++){
                std::string temp_string = layer[i] + "to_CPU_PE_changed";
                GRBVar transition_temp0 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                PE_changed[i].push_back(transition_temp0);
                temp_string = layer[i] + "to_GPU_PE_changed";
                GRBVar transition_temp1 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                PE_changed[i].push_back(transition_temp1);
                temp_string = layer[i] + "to_CPU2_PE_changed";
                GRBVar transition_temp2 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                PE_changed[i].push_back(transition_temp2);
                temp_string = layer[i] + "to_CPU3_PE_changed";
                GRBVar transition_temp3 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                PE_changed[i].push_back(transition_temp3);
                temp_string = layer[i] + "to_CPU4_PE_changed";
                GRBVar transition_temp4 = model.addVar(0.0, 1.0, 0.0, GRB_INTEGER, temp_string);
                PE_changed[i].push_back(transition_temp4);
            }
            //Add max variable
            GRBVar MAX_ob = model.addVar(0.0, 999999999.0, 0.0, GRB_CONTINUOUS, "max (PE)");
            GRBVar min_ob = model.addVar(0.0, 999999999.0, 0.0, GRB_CONTINUOUS, "min (PE)");
            
            GRBLinExpr tempConstrlhs = 0;
            model.addConstr(layer_GPU[0],'=',0, "data_does_not_start_with_GPU");
            model.addConstr(layer_GPU[layer.size()-1],'=',0, "data_does_not_end_with_GPU");
            
            
            // Add constraint: layer_PE[i] = 1 guaruntee node has one processor
            for (int i = 0; i < layer.size(); i ++)   {
                std::string temp_string = layer[i];
                temp_string = temp_string + "_node";
                GRBLinExpr addConstrlhs = 0;
                addConstrlhs += layer_CPU[i] + layer_GPU[i];
                if (layer_CPU2.size())  {
                    addConstrlhs += layer_CPU2[i];
                }
                if (layer_CPU3.size())  {
                    addConstrlhs += layer_CPU3[i];
                }
                if (layer_CPU4.size())  {
                    addConstrlhs += layer_CPU4[i];
                }
                if (layer_npu.size())  {
                    addConstrlhs += layer_npu[i];
                }
                if (layer_dsp.size())  {
                    addConstrlhs += layer_dsp[i];
                }
                model.addConstr(addConstrlhs,'=',1, temp_string);
            }
            
            // Add constraint: 0 = layer0_start_time;
            
            tempConstrlhs = 0;
            tempConstrlhs += layer_start[0];
            model.addConstr(tempConstrlhs,'=',0, "layer0_start_time");
            
            // Add constraint: layer_start+ layer_GPU_n*GPUtim + layer_CPU_n*CPUtime = layer_end;
            for (int i = 0; i < layer.size(); i ++)   {
                GRBLinExpr addConstrlhs = 0;
                addConstrlhs += (layer_start[i] + layer_CPU[i] * CPU_time[i] + layer_GPU[i] * GPU_time[i]);
                if (layer_CPU2.size())  {
                    addConstrlhs += layer_CPU2[i] * CPU2_time[i];
                }
                if (layer_CPU3.size())  {
                    addConstrlhs += layer_CPU3[i] * CPU3_time[i];
                }
                if (layer_CPU4.size())  {
                    addConstrlhs += layer_CPU4[i] * CPU4_time[i];
                }
                if (layer_npu.size())  {
                    addConstrlhs += layer_npu[i] * npu_time[i];
                }
                if (layer_dsp.size())  {
                    addConstrlhs += layer_dsp[i] * dsp_time[i];
                }
                model.addConstr(addConstrlhs,'=',layer_end[i], "layer_end_time");
            }
            
            // Add constraint: post_CPU
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = temp_string0 + "post_CPU";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_CPU[j];
                        model.addConstr(addConstrlhs0,'<',post_CPU[i],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                GRBLinExpr addConstrlhs0 = 0;
                temp_string0 = temp_string0 + "_post_CPU_";
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        addConstrlhs0 += layer_CPU[j];
                    }
                }
                model.addConstr(addConstrlhs0,'>',post_CPU[i],temp_string0);
            }
            // Add constraint: CPU_cluster
            for (int i = 0; i < layer.size()-1; i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "CPU_cluster_start1";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_CPU[i] + (1 - post_CPU[i]) - 1;
                model.addConstr(addConstrlhs0,'<',CPU_cluster[i],temp_string0);
            }
            for (int i = 0; i < layer.size()-1; i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "CPU_cluster_start2";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_CPU[i];
                model.addConstr(addConstrlhs0,'>',CPU_cluster[i],temp_string0);
            }
            for (int i = 0; i < layer.size()-1; i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "CPU_cluster_start3";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += (1 - post_CPU[i]);
                model.addConstr(addConstrlhs0,'>',CPU_cluster[i],temp_string0);
            }
            // Add constraint: CPU_continue
            model.addConstr(layer_CPU[layer_CPU.size()-1] + layer_CPU[0] - 1,'<',CPU_continue,"CPU end-start linked 1");
            model.addConstr(layer_CPU[layer_CPU.size()-1],'>',CPU_continue,"CPU end-start linked 2");
            model.addConstr(layer_CPU[0],'>',CPU_continue,"CPU end-start linked 3");
            //unmap[i][j][0]
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = "CPU1 " + temp_string0 + "unmap1";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_CPU[i] + layer_GPU[j] - 1;
                        model.addConstr(addConstrlhs0,'<',unmap[i][j][0],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = "CPU1 " + temp_string0 + "unmap2";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_CPU[i];
                        model.addConstr(addConstrlhs0,'>',unmap[i][j][0],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = "CPU1 " + temp_string0 + "unmap3";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_GPU[j];
                        model.addConstr(addConstrlhs0,'>',unmap[i][j][0],temp_string0);
                    }
                }
            }
            //Add constraint: memcpy[i][j][0]
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = "CPU1 " + temp_string0 + "memcpy1";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_CPU[i] + (1 - layer_CPU[j]) - 1;
                        model.addConstr(addConstrlhs0,'<',memcpy[i][j][0],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = "CPU1 " + temp_string0 + "memcpy2";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_CPU[i];
                        model.addConstr(addConstrlhs0,'>',memcpy[i][j][0],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = "CPU1 " + temp_string0 + "memcpy3";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += (1 - layer_CPU[j]);
                        model.addConstr(addConstrlhs0,'>',memcpy[i][j][0],temp_string0);
                    }
                }
            }
            //PE_changed[i][0]
            model.addConstr(PE_changed[0][0],'=',0,"PE cannot be changed at start");
            for (int i = 1; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "PE_changed_1";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_CPU[i] + (1 - layer_CPU[i-1]) - 1;
                model.addConstr(addConstrlhs0,'<',PE_changed[i][0],temp_string0);
            }
            for (int i = 1; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "PE_changed_2";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_CPU[i];
                model.addConstr(addConstrlhs0,'>',PE_changed[i][0],temp_string0);
            }
            for (int i = 1; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "PE_changed_3";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += (1 - layer_CPU[i-1]);
                model.addConstr(addConstrlhs0,'>',PE_changed[i][0],temp_string0);
            }
            // Add constraint: post_GPU
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = temp_string0 + "post_GPU";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_GPU[j];
                        model.addConstr(addConstrlhs0,'<',post_GPU[i],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                GRBLinExpr addConstrlhs0 = 0;
                temp_string0 = temp_string0 + "_post_GPU_";
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        addConstrlhs0 += layer_GPU[j];
                    }
                }
                model.addConstr(addConstrlhs0,'>',post_GPU[i],temp_string0);
            }
            // Add constraint: GPU_cluster
            for (int i = 0; i < layer.size()-1; i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "GPU_cluster_start1";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_GPU[i] + (1 - post_GPU[i]) - 1;
                model.addConstr(addConstrlhs0,'<',GPU_cluster[i],temp_string0);
            }
            for (int i = 0; i < layer.size()-1; i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "GPU_cluster_start2";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_GPU[i];
                model.addConstr(addConstrlhs0,'>',GPU_cluster[i],temp_string0);
            }
            for (int i = 0; i < layer.size()-1; i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "GPU_cluster_start3";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += (1 - post_GPU[i]);
                model.addConstr(addConstrlhs0,'>',GPU_cluster[i],temp_string0);
            }
            //map[i][j][0]
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = temp_string0 + "map1";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_GPU[i] + (1-layer_GPU[j]) - 1;
                        model.addConstr(addConstrlhs0,'<',map[i][j][0],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = temp_string0 + "map2";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += layer_GPU[i];
                        model.addConstr(addConstrlhs0,'>',map[i][j][0],temp_string0);
                    }
                }
            }
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = i; j < layer.size(); j ++)    {
                    if (layer_comm[i][j]) {
                        std::string temp_string0 = layer[i];
                        temp_string0 = temp_string0 + "map3";
                        GRBLinExpr addConstrlhs0 = 0;
                        addConstrlhs0 += (1-layer_GPU[j]);
                        model.addConstr(addConstrlhs0,'>',map[i][j][0],temp_string0);
                    }
                }
            }
            //PE_changed[i][1]
            model.addConstr(PE_changed[0][1],'=',0,"PE cannot be changed at start");
            for (int i = 1; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "PE_changed_1";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_GPU[i] + (1-layer_GPU[i-1]) - 1;
                model.addConstr(addConstrlhs0,'<',PE_changed[i][1],temp_string0);
            }
            for (int i = 1; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "PE_changed_2";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += layer_GPU[i];
                model.addConstr(addConstrlhs0,'>',PE_changed[i][1],temp_string0);
            }
            for (int i = 1; i < layer.size(); i ++)   {
                std::string temp_string0 = layer[i];
                temp_string0 = temp_string0 + "PE_changed_3";
                GRBLinExpr addConstrlhs0 = 0;
                addConstrlhs0 += (1-layer_GPU[i-1]);
                model.addConstr(addConstrlhs0,'>',PE_changed[i][1],temp_string0);
            }
            
            if (layer_CPU2.size())  {
                // Add constraint: post_CPU2
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "post_CPU2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU2[j];
                            model.addConstr(addConstrlhs0,'<',post_CPU2[i],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    GRBLinExpr addConstrlhs0 = 0;
                    temp_string0 = temp_string0 + "_post_CPU2_";
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            addConstrlhs0 += layer_CPU2[j];
                        }
                    }
                    model.addConstr(addConstrlhs0,'>',post_CPU2[i],temp_string0);
                }
                // Add constraint: CPU2_cluster
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster2_start1";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU2[i] + (1 - post_CPU2[i]) - 1;
                    model.addConstr(addConstrlhs0,'<',CPU2_cluster[i],temp_string0);
                }
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster2_start2";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU2[i];
                    model.addConstr(addConstrlhs0,'>',CPU2_cluster[i],temp_string0);
                }
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster2_start3";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += (1 - post_CPU2[i]);
                    model.addConstr(addConstrlhs0,'>',CPU2_cluster[i],temp_string0);
                }
                // Add constraint: CPU2_continue
                model.addConstr(layer_CPU2[layer_CPU2.size()-1] + layer_CPU2[0] - 1,'<',CPU2_continue,"CPU2 end-start linked 1");
                model.addConstr(layer_CPU2[layer_CPU2.size()-1],'>',CPU2_continue,"CPU2 end-start linked 2");
                model.addConstr(layer_CPU2[0],'>',CPU2_continue,"CPU2 end-start linked 3");
                //unmap[j][i][1]
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[j][i]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU2_map_1";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU2[i] + layer_GPU[j] - 1;
                            model.addConstr(addConstrlhs0,'<',unmap[j][i][1],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU2_map_2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU2[i];
                            model.addConstr(addConstrlhs0,'>',unmap[i][j][1],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU2_map_3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_GPU[j];
                            model.addConstr(addConstrlhs0,'>',unmap[i][j][1],temp_string0);
                        }
                    }
                }
                //Add constraint: memcpy[i][j][1]
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU2 " + temp_string0 + "memcpy1";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU2[i] + (1 - layer_CPU2[j]) - 1;
                            model.addConstr(addConstrlhs0,'<',memcpy[i][j][1],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU2 " + temp_string0 + "memcpy2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU2[i];
                            model.addConstr(addConstrlhs0,'>',memcpy[i][j][1],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU2 " + temp_string0 + "memcpy3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += (1 - layer_CPU2[j]);
                            model.addConstr(addConstrlhs0,'>',memcpy[i][j][1],temp_string0);
                        }
                    }
                }
                //PE_changed[i][2]
                model.addConstr(PE_changed[0][2],'=',0,"PE cannot be changed at start");
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_1";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU2[i] + (1 - layer_CPU2[i-1]) - 1;
                    model.addConstr(addConstrlhs0,'<',PE_changed[i][2],temp_string0);
                }
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_2";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU2[i];
                    model.addConstr(addConstrlhs0,'>',PE_changed[i][2],temp_string0);
                }
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_3";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += (1 - layer_CPU2[i-1]);
                    model.addConstr(addConstrlhs0,'>',PE_changed[i][2],temp_string0);
                }
            }
            
            if (layer_CPU3.size())  {
                // Add constraint: post_CPU3
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "post_CPU3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU3[j];
                            model.addConstr(addConstrlhs0,'<',post_CPU3[i],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    GRBLinExpr addConstrlhs0 = 0;
                    temp_string0 = temp_string0 + "_post_CPU3_";
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            addConstrlhs0 += layer_CPU3[j];
                        }
                    }
                    model.addConstr(addConstrlhs0,'>',post_CPU3[i],temp_string0);
                }
                // Add constraint: CPU3_cluster
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster3_start1";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU3[i] + (1 - post_CPU3[i]) - 1;
                    model.addConstr(addConstrlhs0,'<',CPU3_cluster[i],temp_string0);
                }
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster3_start2";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU3[i];
                    model.addConstr(addConstrlhs0,'>',CPU3_cluster[i],temp_string0);
                }
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster3_start3";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += (1 - post_CPU3[i]);
                    model.addConstr(addConstrlhs0,'>',CPU3_cluster[i],temp_string0);
                }
                // Add constraint: CPU3_continue
                model.addConstr(layer_CPU3[layer_CPU3.size()-1] + layer_CPU3[0] - 1,'<',CPU3_continue,"CPU3 end-start linked 1");
                model.addConstr(layer_CPU3[layer_CPU3.size()-1],'>',CPU3_continue,"CPU3 end-start linked 2");
                model.addConstr(layer_CPU3[0],'>',CPU3_continue,"CPU3 end-start linked 3");
                //unmap[j][i][2]
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU3_map_1";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU3[i] + layer_GPU[j] - 1;
                            model.addConstr(addConstrlhs0,'<',unmap[i][j][2],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU3_map_2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU3[i];
                            model.addConstr(addConstrlhs0,'>',unmap[i][j][2],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU3_map_3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_GPU[j];
                            model.addConstr(addConstrlhs0,'>',unmap[i][j][2],temp_string0);
                        }
                    }
                }
                //Add constraint: memcpy[i][j][2]
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU3 " + temp_string0 + "memcpy1";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU3[i] + (1 - layer_CPU3[j]) - 1;
                            model.addConstr(addConstrlhs0,'<',memcpy[i][j][2],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU3 " + temp_string0 + "memcpy2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU3[i];
                            model.addConstr(addConstrlhs0,'>',memcpy[i][j][2],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU3 " + temp_string0 + "memcpy3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += (1 - layer_CPU3[j]);
                            model.addConstr(addConstrlhs0,'>',memcpy[i][j][2],temp_string0);
                        }
                    }
                }
                //PE_changed[i][3]
                model.addConstr(PE_changed[0][3],'=',0,"PE cannot be changed at start");
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_1";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU3[i] + (1 - layer_CPU3[i-1]) - 1;
                    model.addConstr(addConstrlhs0,'<',PE_changed[i][3],temp_string0);
                }
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_2";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU3[i];
                    model.addConstr(addConstrlhs0,'>',PE_changed[i][3],temp_string0);
                }
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_3";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += (1 - layer_CPU3[i-1]);
                    model.addConstr(addConstrlhs0,'>',PE_changed[i][3],temp_string0);
                }
            }
            
            if (layer_CPU4.size())  {
                // Add constraint: post_CPU4
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "post_CPU4";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU4[j];
                            model.addConstr(addConstrlhs0,'<',post_CPU4[i],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    GRBLinExpr addConstrlhs0 = 0;
                    temp_string0 = temp_string0 + "_post_CPU4_";
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            addConstrlhs0 += layer_CPU4[j];
                        }
                    }
                    model.addConstr(addConstrlhs0,'>',post_CPU4[i],temp_string0);
                }
                // Add constraint: CPU4_cluster
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster4_start1";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU4[i] + (1 - post_CPU4[i]) - 1;
                    model.addConstr(addConstrlhs0,'<',CPU4_cluster[i],temp_string0);
                }
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster4_start2";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU4[i];
                    model.addConstr(addConstrlhs0,'>',CPU4_cluster[i],temp_string0);
                }
                for (int i = 0; i < layer.size()-1; i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "CPU_cluster4_start3";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += (1 - post_CPU4[i]);
                    model.addConstr(addConstrlhs0,'>',CPU4_cluster[i],temp_string0);
                }
                // Add constraint: CPU4_continue
                model.addConstr(layer_CPU4[layer_CPU4.size()-1] + layer_CPU4[0] - 1,'<',CPU4_continue,"CPU4 end-start linked 1");
                model.addConstr(layer_CPU4[layer_CPU4.size()-1],'>',CPU4_continue,"CPU4 end-start linked 2");
                model.addConstr(layer_CPU4[0],'>',CPU4_continue,"CPU4 end-start linked 3");
                //unmap[j][i][3]
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU4_comm2_1";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU4[i] + layer_GPU[j] - 1;
                            model.addConstr(addConstrlhs0,'<',unmap[i][j][3],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU4_comm2_2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU4[i];
                            model.addConstr(addConstrlhs0,'>',unmap[i][j][3],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = temp_string0 + "to_CPU4_comm2_3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_GPU[j];
                            model.addConstr(addConstrlhs0,'>',unmap[i][j][3],temp_string0);
                        }
                    }
                }
                //Add constraint: memcpy[i][j][3]
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU4 " + temp_string0 + "memcpy1";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU4[i] + (1 - layer_CPU2[j]) - 1;
                            model.addConstr(addConstrlhs0,'<',memcpy[i][j][3],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU4 " + temp_string0 + "memcpy2";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += layer_CPU4[i];
                            model.addConstr(addConstrlhs0,'>',memcpy[i][j][3],temp_string0);
                        }
                    }
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)    {
                        if (layer_comm[i][j]) {
                            std::string temp_string0 = layer[i];
                            temp_string0 = "CPU4 " + temp_string0 + "memcpy3";
                            GRBLinExpr addConstrlhs0 = 0;
                            addConstrlhs0 += (1 - layer_CPU4[j]);
                            model.addConstr(addConstrlhs0,'>',memcpy[i][j][3],temp_string0);
                        }
                    }
                }
                //PE_changed[i][4]
                model.addConstr(PE_changed[0][4],'=',0,"PE cannot be changed at start");
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_1";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU4[i] + (1 - layer_CPU4[i-1]) - 1;
                    model.addConstr(addConstrlhs0,'<',PE_changed[i][4],temp_string0);
                }
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_2";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += layer_CPU4[i];
                    model.addConstr(addConstrlhs0,'>',PE_changed[i][4],temp_string0);
                }
                for (int i = 1; i < layer.size(); i ++)   {
                    std::string temp_string0 = layer[i];
                    temp_string0 = temp_string0 + "PE_changed_3";
                    GRBLinExpr addConstrlhs0 = 0;
                    addConstrlhs0 += (1 - layer_CPU4[i-1]);
                    model.addConstr(addConstrlhs0,'>',PE_changed[i][4],temp_string0);
                }
            }
            
            //last layer cluster setting
            tempConstrlhs = 0;
            tempConstrlhs += layer_CPU[layer.size()-1] - CPU_continue;
            model.addConstr(tempConstrlhs,'=',CPU_cluster[layer.size()-1],"end CPU cluster");
            
            if (layer_CPU2.size())  {
                tempConstrlhs = 0;
                tempConstrlhs += layer_CPU2[layer.size()-1] - CPU2_continue;
                model.addConstr(tempConstrlhs,'=',CPU2_cluster[layer.size()-1],"end CPU2 cluster");
            }
            if (layer_CPU3.size())  {
                tempConstrlhs = 0;
                tempConstrlhs += layer_CPU3[layer.size()-1] - CPU3_continue;
                model.addConstr(tempConstrlhs,'=',CPU3_cluster[layer.size()-1],"end CPU3 cluster");
            }
            if (layer_CPU4.size())  {
                tempConstrlhs = 0;
                tempConstrlhs += layer_CPU4[layer.size()-1] - CPU4_continue;
                model.addConstr(tempConstrlhs,'=',CPU4_cluster[layer.size()-1],"end CPU4 cluster");
            }
            
            //CPU cluster numbers
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += CPU_cluster[i];
            }
            model.addConstr(tempConstrlhs,'=',1,"one CPU cluster");
            if (layer_CPU2.size())  {
                tempConstrlhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs += CPU2_cluster[i];
                }
                model.addConstr(tempConstrlhs,'=',1,"one CPU2 cluster");
            }
            if (layer_CPU3.size())  {
                tempConstrlhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs += CPU3_cluster[i];
                }
                model.addConstr(tempConstrlhs,'=',1,"one CPU3 cluster");
            }
            if (layer_CPU4.size())  {
                tempConstrlhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs += CPU4_cluster[i];
                }
                model.addConstr(tempConstrlhs,'=',1,"one CPU4 cluster");
            }
            //only one GPU cluster
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += GPU_cluster[i];
            }
            model.addConstr(tempConstrlhs,'=',1,"one GPU cluster");
            
            //PE_change numbers
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += PE_changed[i][0];
                tempConstrlhs += PE_changed[i][1];
                tempConstrlhs += PE_changed[i][2];
                tempConstrlhs += PE_changed[i][3];
                tempConstrlhs += PE_changed[i][4];
            }
            model.addConstr(tempConstrlhs,'<',num_of_PE,"total PE change");
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += PE_changed[i][0];
            }
            model.addConstr(tempConstrlhs,'<',1,"total PE change");
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += PE_changed[i][1];
            }
            model.addConstr(tempConstrlhs,'<',1,"total PE change");
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += PE_changed[i][2];
            }
            model.addConstr(tempConstrlhs,'<',1,"total PE change");
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += PE_changed[i][3];
            }
            model.addConstr(tempConstrlhs,'<',1,"total PE change");
            tempConstrlhs = 0;
            for (int i = 0; i < layer.size(); i ++)   {
                tempConstrlhs += PE_changed[i][4];
            }
            model.addConstr(tempConstrlhs,'<',1,"total PE change");
            
            // Add constraint: ijCPU
            for (int i = 1; i < layer.size(); i ++)   {
                for (int j = 0; j < layer.size(); j ++)   {
                    std::string temp_string = layer[j];
                    temp_string = temp_string + "_and_" + layer[i] + "_are_CPU_mapped1";
                    GRBLinExpr addConstrlhs = 0;
                    addConstrlhs += layer_CPU[i] + layer_CPU[j] - 1;
                    model.addConstr(addConstrlhs,'<',same[i][j][1], temp_string);
                    
                    temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU_mapped2";
                    addConstrlhs = 0;
                    addConstrlhs += layer_CPU[i];
                    model.addConstr(addConstrlhs,'>',same[i][j][1], temp_string);
                    
                    temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU_mapped3";
                    addConstrlhs = 0;
                    addConstrlhs += layer_CPU[j];
                    model.addConstr(addConstrlhs,'>',same[i][j][1], temp_string);
                    
                    temp_string = layer[j] + "_and_" + layer[i] + "_are_GPU_mapped1";
                    addConstrlhs = 0;
                    addConstrlhs += layer_GPU[i] + layer_GPU[j] - 1;
                    model.addConstr(addConstrlhs,'<',same[i][j][2], temp_string);
                    
                    temp_string = layer[j] + "_and_" + layer[i] + "_are_GPU_mapped2";
                    addConstrlhs = 0;
                    addConstrlhs += layer_GPU[i];
                    model.addConstr(addConstrlhs,'>',same[i][j][2], temp_string);
                    
                    temp_string = layer[j] + "_and_" + layer[i] + "_are_GPU_mapped3";
                    addConstrlhs = 0;
                    addConstrlhs += layer_GPU[j];
                    model.addConstr(addConstrlhs,'>',same[i][j][2], temp_string);
                    
                    if (layer_CPU2.size())  {
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU2_mapped1";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU2[i] + layer_CPU2[j] - 1;
                        model.addConstr(addConstrlhs,'<',same[i][j][3], temp_string);
                        
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU2_mapped2";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU2[i];
                        model.addConstr(addConstrlhs,'>',same[i][j][3], temp_string);
                        
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU2_mapped3";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU2[j];
                        model.addConstr(addConstrlhs,'>',same[i][j][3], temp_string);
                    }
                    if (layer_CPU3.size())  {
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU3_mapped1";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU3[i] + layer_CPU3[j] - 1;
                        model.addConstr(addConstrlhs,'<',same[i][j][4], temp_string);
                        
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU3_mapped2";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU3[i];
                        model.addConstr(addConstrlhs,'>',same[i][j][4], temp_string);
                        
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU3_mapped3";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU3[j];
                        model.addConstr(addConstrlhs,'>',same[i][j][4], temp_string);
                    }
                    if (layer_CPU4.size())  {
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU4_mapped1";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU4[i] + layer_CPU4[j] - 1;
                        model.addConstr(addConstrlhs,'<',same[i][j][5], temp_string);
                        
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU4_mapped2";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU4[i];
                        model.addConstr(addConstrlhs,'>',same[i][j][5], temp_string);
                        
                        temp_string = layer[j] + "_and_" + layer[i] + "_are_CPU4_mapped3";
                        addConstrlhs = 0;
                        addConstrlhs += layer_CPU4[j];
                        model.addConstr(addConstrlhs,'>',same[i][j][5], temp_string);
                    }
                    
                    temp_string = temp_string + "_and_" + layer[i] + "_are_same_PE";
                    addConstrlhs = 0;
                    addConstrlhs += same[i][j][1] + same[i][j][2];
                    if (layer_CPU2.size())  {
                        addConstrlhs +=  same[i][j][3];
                    }
                    if (layer_CPU3.size())  {
                        addConstrlhs +=  same[i][j][4];
                    }
                    if (layer_CPU4.size())  {
                        addConstrlhs +=  same[i][j][5];
                    }
                    model.addConstr(addConstrlhs,'=',same[i][j][0], temp_string);
                }
            }
            
            // Add constraint: i,j is different or not? if mapped same processor, start time is greater than end time.
            for (int i = 0; i < layer.size(); i ++)   {
                for (int j = 0; j < i; j ++)   {
                    std::string temp_string = layer[j];
                    temp_string = temp_string + "_are_mapped_same_place_with_" + layer[i];
                    GRBLinExpr addConstrlhs = 0;
                    addConstrlhs += layer_start[i] - layer_end[j] +((1-same[j][i][0]) * 4999999) ;
                    model.addConstr(addConstrlhs,'>',0, temp_string);
                }
            }
            
            
            if (mode) {
                //set pipeline stage
                tempConstrlhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs += CPU_time[i] * layer_CPU[i];
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)   {
                        tempConstrlhs += unmap[i][j][0] * (unmap_alpha * layer_comm[i][j] +unmap_beta);
                        tempConstrlhs += memcpy[i][j][0] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma);
                    }
                }
                model.addConstr(tempConstrlhs,'<',MAX_ob, "CPU pipeline");
                
                tempConstrlhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs += GPU_time[i] * layer_GPU[i];
                }
                for (int i = 0; i < layer.size(); i ++)   {
                    for (int j = i; j < layer.size(); j ++)   {
                        tempConstrlhs += map[i][j][0] * (map_alpha * layer_comm[i][j] +map_beta);
                        tempConstrlhs += map[i][j][0] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma);
                    }
                }
                model.addConstr(tempConstrlhs,'<',MAX_ob, "GPU pipeline");
                
                if (layer_CPU2.size())  {
                    tempConstrlhs = 0;
                    for (int i = 0; i < layer.size(); i ++)   {
                        tempConstrlhs += CPU2_time[i] * layer_CPU2[i];
                    }
                    for (int i = 0; i < layer.size(); i ++)   {
                        for (int j = i; j < layer.size(); j ++)   {
                            tempConstrlhs += unmap[i][j][1] * (unmap_alpha * layer_comm[i][j] +unmap_beta);
                            tempConstrlhs += memcpy[i][j][1] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma);
                        }
                    }
                    model.addConstr(tempConstrlhs,'<',MAX_ob, "CPU2 pipeline");
                }
                if (layer_CPU3.size())  {
                    tempConstrlhs = 0;
                    for (int i = 0; i < layer.size(); i ++)   {
                        tempConstrlhs += CPU3_time[i] * layer_CPU3[i];
                    }
                    for (int i = 0; i < layer.size(); i ++)   {
                        for (int j = i; j < layer.size(); j ++)   {
                            tempConstrlhs += unmap[i][j][2] * (unmap_alpha * layer_comm[i][j] +unmap_beta);
                            tempConstrlhs += memcpy[i][j][2] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma);
                        }
                    }
                    model.addConstr(tempConstrlhs,'<',MAX_ob, "CPU3 pipeline");
                }
                if (layer_CPU4.size())  {
                    tempConstrlhs = 0;
                    for (int i = 0; i < layer.size(); i ++)   {
                        tempConstrlhs += CPU4_time[i] * layer_CPU4[i];
                    }
                    for (int i = 0; i < layer.size(); i ++)   {
                        for (int j = i; j < layer.size(); j ++)   {
                            tempConstrlhs += unmap[i][j][3] * (unmap_alpha * layer_comm[i][j] +unmap_beta);
                            tempConstrlhs += memcpy[i][j][3] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma);
                        }
                    }
                    model.addConstr(tempConstrlhs,'<',MAX_ob, "CPU4 pipeline");
                }
                
                //set whole CPU util
                tempConstrlhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs += CPU_time[i] * layer_CPU[i] * num_of_CPU_core;
                    for (int j = i; j < layer.size(); j ++)   {
                        tempConstrlhs += unmap[i][j][0] * (unmap_alpha * layer_comm[i][j] +unmap_beta) * num_of_CPU_core;
                        tempConstrlhs += memcpy[i][j][0] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma)* num_of_CPU_core;
                    }
                    if (layer_CPU2.size())  {
                            tempConstrlhs += CPU2_time[i] * layer_CPU2[i] * num_of_CPU2_core;
                        for (int j = i; j < layer.size(); j ++)   {
                            tempConstrlhs += unmap[i][j][1] * (unmap_alpha * layer_comm[i][j] +unmap_beta) * num_of_CPU2_core;
                            tempConstrlhs += memcpy[i][j][1] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma) * num_of_CPU2_core;
                        }
                    }
                    if (layer_CPU3.size())  {
                            tempConstrlhs += CPU3_time[i] * layer_CPU3[i] * num_of_CPU3_core;
                        for (int j = i; j < layer.size(); j ++)   {
                            tempConstrlhs += unmap[i][j][2] * (unmap_alpha * layer_comm[i][j] +unmap_beta) * num_of_CPU3_core;
                            tempConstrlhs += memcpy[i][j][1] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma) * num_of_CPU3_core;
                        }
                    }
                    if (layer_CPU4.size())  {
                        tempConstrlhs += CPU4_time[i] * layer_CPU4[i] * num_of_CPU4_core;
                        for (int j = i; j < layer.size(); j ++)   {
                            tempConstrlhs += unmap[i][j][3] * (unmap_alpha * layer_comm[i][j] +unmap_beta) * num_of_CPU4_core;
                            tempConstrlhs += memcpy[i][j][3] * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma) * num_of_CPU4_core;
                        }
                    }
                }
                tempConstrlhs *= 25;
                
                tempConstrlhs /= CPU_utilization;
                
                
                for (int i = 0; i < layer.size(); i ++)   {
                    tempConstrlhs -= GPU_time[i] * layer_GPU[i];
                }
                
                model.addConstr(tempConstrlhs,'<',0, "Whole CPU Utilization");
            }
            
            // Set objective: maximize throughput?
            
            if (mode) {
                
                GRBLinExpr lhs = MAX_ob;
                model.setObjective(lhs, GRB_MINIMIZE);
            }
            
            // Set objective: minimize last layer's end time
            else {
                GRBLinExpr lhs = 0;
                for (int i = 0; i < layer.size(); i ++)   {
                    lhs += layer_end[i];
                }
                model.setObjective(lhs, GRB_MINIMIZE);
            }
            
            // Optimize model
            model.optimize();
            data_for_merging = new int[layer.size()];
            
            //result output
            if (mode)   {
                throughput = MAX_ob.get(GRB_DoubleAttr_X);
                // std::cout << "Obj: MAX " << throughput << std::endl;
            }
            // std::cout << "Obj: end time " << layer_end[layer.size()-1].get(GRB_DoubleAttr_X) << std::endl;
            
            // struct
            std::vector<result> temp_result_set;
            for (int i = 0; i < layer.size(); i ++)   {
                result temp_result;
                temp_result.layer_name = layer[i];
                int temp_PE_number = 0;
                if ((long)(layer_CPU[i].get(GRB_DoubleAttr_X)+0.5))   {
                    temp_result.mapped = "cpu";
                    data_for_merging[i]= 0;
                }
                else  {
                    temp_result.mapped = "gpu";
                    data_for_merging[i]= 1;
                }
                if (layer_CPU2.size()) {
                    if ((long)(layer_CPU[i].get(GRB_DoubleAttr_X)+0.5))   {
                        temp_result.mapped = "cpu";
                        data_for_merging[i]= 0;
                    }
                    else if ((long)(layer_CPU2[i].get(GRB_DoubleAttr_X)+0.5))  {
                        temp_result.mapped = "cpu2";
                        data_for_merging[i]= 1;
                    }
                    else  {
                        temp_result.mapped = "gpu";
                        data_for_merging[i]= 2;
                    }
                }
                if (layer_CPU3.size()) {
                    if ((long)(layer_CPU[i].get(GRB_DoubleAttr_X)+0.5))   {
                        temp_result.mapped = "cpu";
                        data_for_merging[i]= 0;
                    }
                    else if ((long)(layer_CPU2[i].get(GRB_DoubleAttr_X)+0.5))  {
                        temp_result.mapped = "cpu2";
                        data_for_merging[i]= 1;
                    }
                    else if ((long)(layer_CPU3[i].get(GRB_DoubleAttr_X)+0.5))  {
                        temp_result.mapped = "cpu3";
                        data_for_merging[i]= 2;
                    }
                    else  {
                        temp_result.mapped = "gpu";
                        data_for_merging[i]= 3;
                    }
                }
                if (layer_CPU4.size()) {
                    if ((long)(layer_CPU[i].get(GRB_DoubleAttr_X)+0.5))   {
                        temp_result.mapped = "cpu";
                        data_for_merging[i]= 0;
                    }
                    else if ((long)(layer_CPU2[i].get(GRB_DoubleAttr_X)+0.5))  {
                        temp_result.mapped = "cpu2";
                        data_for_merging[i]= 1;
                    }
                    else if ((long)(layer_CPU3[i].get(GRB_DoubleAttr_X)+0.5))  {
                        temp_result.mapped = "cpu3";
                        data_for_merging[i]= 2;
                    }
                    else if ((long)(layer_CPU4[i].get(GRB_DoubleAttr_X)+0.5))  {
                        temp_result.mapped = "cpu4";
                        data_for_merging[i]= 3;
                    }
                    else  {
                        temp_result.mapped = "gpu";
                        data_for_merging[i]= 4;
                    }
                }
                temp_result.comm_time = 0;
                for (int j = i; j < layer.size(); j ++ )    {
                    temp_result.comm_time += (unmap[i][j][0].get(GRB_DoubleAttr_X) * (unmap_alpha * layer_comm[i][j] +unmap_beta));
                    temp_result.comm_time += (memcpy[i][j][0].get(GRB_DoubleAttr_X) * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma));
                    temp_result.comm_time += (unmap[i][j][1].get(GRB_DoubleAttr_X) * (unmap_alpha * layer_comm[i][j] +unmap_beta));
                    temp_result.comm_time += (memcpy[i][j][1].get(GRB_DoubleAttr_X) * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma));
                    temp_result.comm_time += (unmap[i][j][2].get(GRB_DoubleAttr_X) * (unmap_alpha * layer_comm[i][j] +unmap_beta));
                    temp_result.comm_time += (memcpy[i][j][2].get(GRB_DoubleAttr_X) * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma));
                    temp_result.comm_time += (unmap[i][j][3].get(GRB_DoubleAttr_X) * (unmap_alpha * layer_comm[i][j] +unmap_beta));
                    temp_result.comm_time += (memcpy[i][j][3].get(GRB_DoubleAttr_X) * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma));
                    temp_result.comm_time += (map[i][j][0].get(GRB_DoubleAttr_X) * (map_alpha * layer_comm[i][j] + map_beta));
                    temp_result.comm_time += (map[i][j][0].get(GRB_DoubleAttr_X) * (memcpy_alpha * (layer_comm[i][j] * layer_comm[i][j]) * memcpy_beta * layer_comm[i][j] + memcpy_gamma));
                }
                
                
                if (temp_result.comm_time < 0.5)    {
                    temp_result.comm_time = 0.0f;
                }
                
                
                if (i == 0)  {
                    temp_result.start_time =  0;
                }
                else    {
                    temp_result.start_time =  temp_result_set[i-1].end_time + temp_result.comm_time;
                }
                temp_result.end_time = temp_result.start_time;
                
                if (temp_result.mapped == "cpu")  {
                    temp_result.end_time += CPU_time[i];
                }
                if (temp_result.mapped == "cpu2")  {
                    temp_result.end_time += CPU2_time[i];
                }
                if (temp_result.mapped == "cpu3")  {
                    temp_result.end_time += CPU3_time[i];
                }
                if (temp_result.mapped == "cpu4")  {
                    temp_result.end_time += CPU4_time[i];
                }
                if (temp_result.mapped == "gpu")  {
                    temp_result.end_time += GPU_time[i];
                }
                
                temp_result_set.push_back(temp_result);
            }
            result_set.push_back(temp_result_set);
        } // try
        catch (GRBException e)   {
            std::cout << "Error code = " << e.getErrorCode() << std::endl;
            std::cout << e.getMessage() << std::endl;
        }
        catch (...)  {
            std::cout << "Exception during optimization" <<  std::endl;
        }
        
        return_result(result_set[0]);
        
        // Debugging
        // for (int i = 0; i < layer.size(); i ++)
        //     std::cout <<result_set[0][i].layer_name <<'\t' <<result_set[0][i].mapped <<'\t' <<result_set[0][i].start_time <<'\t' <<result_set[0][i].end_time <<'\t' <<result_set[0][i].comm_time <<std::endl;
        // for (int i = 0; i < layer.size(); i ++)
        //     std::cout << i << "th layer mapping number is " << data_for_merging[i]  <<std::endl;

        // For Python porting -----------------------------
        num_layer = layer.size();
        for (int i = 0; i < num_layer; i++)
        {
            data.layer_name = result_set[0][i].layer_name;
            data.mapped = result_set[0][i].mapped;
            data.start_time = result_set[0][i].start_time;
            data.end_time = result_set[0][i].end_time;
            data.comm_time = result_set[0][i].comm_time;
            
            res.push_back(data); // All information will be here (global variable)
        }
        
        delete(caffe_model);
        delete(time);

        return 0;
    }

	// For Python porting --------------------------------
	char* get_layer_name_result(int vector_index)
	{
		std::vector<char> writable(res[vector_index].layer_name.begin(), res[vector_index].layer_name.end());
		writable.push_back('\0');
		global_writable = writable;
		ptr = &global_writable[0];
		return ptr;
	}

	char* get_mapped_result(int vector_index)
	{
		std::vector<char> writable(res[vector_index].mapped.begin(), res[vector_index].mapped.end());
		writable.push_back('\0');
		global_writable = writable;
		ptr = &global_writable[0];
		return ptr;
	}

	float get_start_time_result(int vector_index)
	{
		return res[vector_index].start_time;
	}

	float get_end_time_result(int vector_index)
	{
		return res[vector_index].end_time;
	}

	float get_comm_time_result(int vector_index)
	{
		return res[vector_index].comm_time;
	}

    int get_mapping(int index) {
        return data_for_merging[index];
    }

    int get_throughput() {
        return throughput;
    }

	int get_num_layer(void)
	{
		return num_layer;
	}
}
