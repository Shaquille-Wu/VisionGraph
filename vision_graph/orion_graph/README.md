## vision_graph的编译

在vision_graph根目录下配置了一个自动编译的脚本:build_graph.sh，调用它即可完成vision_graph的编译工作  
1.进入vision_graph根目录: cd ./vision_graph  
2.执行build_graph.sh，假如要编译基于x86_64处理器，linux操作系统的，并且是Debug版的libvision_graph.so,，可以这样：  
```
./build_graph.sh -c build -a x86_64 -o linux -t Debug       //编译
./build_graph.sh -c test -a x86_64 -o linux -t Debug        //执行test
./build_graph.sh -c clean -a x86_64 -o linux -t Debug       //clean掉生成的各种目标文件
```
执行完成后可以在这个目录找到编译生成的文件:./build/COMPILE_TYPE/PLATFORM/vision_graph  
假如要编译基于armv8处理器，android操作系统的，并且是Release版的libvision_graph.so,，可以这样：  
```
./build_graph.sh -c build -a armv8 -o android -t Release       //编译
./build_graph.sh -c clean -a armv8 -o android -t Release       //clean掉生成的各种目标文件
```
3.build_graph.sh的格式: ./build_graph.sh -c CMD -a ARCH -o OS -t COMPILE_TYPE  
CMD: build, test, clean  
ARCH: x86_64, armv8
OS:   linux, android
COMPILE_TYPE:Debug, Release  
以上要注意区分大小写  
其中目前测试有效的有-a armv8 -o android和-a x86_64 -o linux 

4.Android版本编译前需要配置cmake的toolchain，须在脚本中配置NDK的路径，比如
```
ANDROID_NDK_DIR=/home/shaquille/Android/Sdk/ndk-bundle
```
需要根据实际情况配置这个变量 
如果host机已经设定了环境变量"NDK_ROOT"，则无需填写ANDROID_NDK_DIR，脚本将以host的NDK_ROOT为准进行编译

## vision_graph的环境依赖

1.TBB,    TBB的全部代码在vision_graph目录下的oneTBB目录，随vision_graph一起编译  
2.dlcv,   在deps/dlcv目录下, libdlcv.a已经编译好，其android版的依赖环境是android ndk-r17c，其opencv的环境是opencv-4.2.0  
3.opencv, x86_64版的opencv依赖，这里不做说明。需要说明的是若是android版，需要在build_graph.sh中指定opencv的路径，比如
```
ANDROID_OPENCV_DIR_PREFIX=/home/shaquille/WorkSpace/OpenCV-android-sdk/OpenCV-4.2.0-android-sdk/sdk/native/jni
```
4.libtbbmalloc_proxy.so，是TBB生成的3个.so之一，在vision_graph运行之前需要设定环境变量LD_PRELOAD：
```
export LD_PRELOAD=libtbbmalloc_proxy.so
```
在x86_64上运行时，如果没有设定，目前没有发现异常  
在android上运行时，如果没有设定，发现snpe在某些情况下不正常，因此android上，以命令行方式运行时需要设定这个环境变量  
若是android_studio开发的app，需要在java程序加载jni的so之前加载libtbbmalloc_proxy.so，比如:
```
	System.loadLibrary("tbbmalloc_proxy");          //加载libtbbmalloc_proxy.so
	System.loadLibrary("vision_graph_interface");   //加载jni的so
```

## vision_graph的程序开发基本流程

1.声明"vision_graph::Graph"，比如
```
vision_graph::Graph   face_graph;
```

2.加载json配置文件，比如
```
face_graph.Build(graph_json_file);
```

3.获取各个tensor_out，为后续的工作准备数据，比如
```
//通过GetTensors, 先获取各个tensor_out的指针,0表示sub_graph的索引号
std::map<std::string, std::pair<vision_graph::Tensor*, vision_graph::Solver*> >   tensor_map = face_graph.GetTensors(0);
vision_graph::Tensor*            tensor_base_image       = ((tensor_map.find("image"))->second).first;
vision_graph::Tensor*            tensor_base_pos         = ((tensor_map.find("box_selector"))->second).first;
vision_graph::Tensor*            tensor_base_keypoint    = ((tensor_map.find("face_keypoint"))->second).first;
vision_graph::Tensor*            tensor_base_hog         = ((tensor_map.find("face_hog_similarity"))->second).first;
vision_graph::Tensor*            tensor_base_kpts_offset = ((tensor_map.find("face_kpts_offset"))->second).first;
vision_graph::Tensor*            tensor_base_box         = ((tensor_map.find("face_box_smooth"))->second).first;
vision_graph::TensorImage*       tensor_image            = dynamic_cast<vision_graph::TensorImage*>(tensor_base_image);
vision_graph::TensorReference*   tensor_reference        = dynamic_cast<vision_graph::TensorReference*>(tensor_base_pos);
vision_graph::TensorKeypoints*   tensor_keypoints        = dynamic_cast<vision_graph::TensorKeypoints*>(tensor_base_keypoint);
vision_graph::TensorFloat32*     tensor_float32          = dynamic_cast<vision_graph::TensorFloat32*>(tensor_base_hog);
vision_graph::TensorKeypoints*   tensor_kpts_offset      = dynamic_cast<vision_graph::TensorKeypoints*>(tensor_base_kpts_offset);
vision_graph::TensorBox*         tensor_smooth_box       = dynamic_cast<vision_graph::TensorBox*>(tensor_base_box);

//对输入数据进行赋值，比如读取图像
*tensor_image = cv::imread(test_image);
```

4.开始运行，并等待结束，比如
```
face_graph.Start(0); 
face_graph.Wait(0);
```

5.获取tensor_out，由于在build之后已经获取了各个tensor_out的指针，这里无需再调用"GetTensors"，直接使用各个指针的结果即可，比如
```
cv::Rect rect((*tensor_select_box).x1, (*tensor_select_box).y1, (*tensor_select_box).x2-(*tensor_select_box).x1, (*tensor_select_box).y2-(*tensor_select_box).y1);
cv::rectangle(*(tensor_image), rect, cv::Scalar(255, 0, 0), 2, 8);

cv::Rect rect_smooth(FLT2INT32((*tensor_smooth_box).x1), 
					 FLT2INT32((*tensor_smooth_box).y1), 
					 FLT2INT32((*tensor_smooth_box).x2 - (*tensor_smooth_box).x1), 
					 FLT2INT32((*tensor_smooth_box).y2 - (*tensor_smooth_box).y1));
cv::rectangle(*(tensor_image), rect_smooth, cv::Scalar(0, 0, 255), 2, 8);

int  kpt_size = (int)(tensor_kpts_offset->size());
for(int i = 0 ; i < kpt_size ; i ++)
{
	cv::Point2f   cur_pt((*tensor_kpts_offset)[i].x, (*tensor_kpts_offset)[i].y);
	cv::circle(*tensor_image, cur_pt, 2, cv::Scalar(0, 255, 0), -1);
}
```

6.整个程序结束后释放资源，比如
```
face_graph.Destroy();
```

7.至此整个流程介绍完毕，具体可参看demos目录下的face_graph.cpp文件  

## vision_graph的继承与扩展  
1.graph，从组合的概念来看，包括: graph->sub_graph->node->solver  
2.solver对tensor进行操作  
3.graph对用户可见的部分只有solver和tensor，因此用户的继承与扩展也是围绕这两个类来进行  
4.solver的定义决定了node的定义，node只负责数据的传递，属于graph的内部功能，并不负责运算。同时，用户关心的是solver的具体定义和行为，因此用户只需继承和扩展solver即可  
5.graph内部只定义了有限的solver和tensor，对于具体应用而言，用户需要根据具体情况自行定义  
6.solver类的注册，graph需要知道solver类名到create(solver)函数的隐射，graph才能在build阶段创建出solver实例，  
  因此solver的定义需要通过组合继承的方式，继承Solver和vision_graph::SolverCreator<SolverXXX>，这样才能完成这个扩展类的注册，比如
```
class SolverDetect : public Solver, vision_graph::SolverCreator<SolverDetect>
```
7.Tensor的扩展，由于Tensor的Create都是通过solver来完成的，因此，扩展的Tensor无需像Solver一样需要注册，用户只需继承Tensor，并定义相应的tensor_type即可，即实现相应的GetType()函数  

## solver的分类  
1.intuitive_solvers，这部分为graph基本节点提供支撑，比如数据、条件、跳转、选择、逻辑操作等等，属于graph本身要用到的solver  
2.function_solvers，这部分是具体应用需要用的solver，比如检测、跟踪、分类、识别等等，属于具体的功能单元  
3.这两部分对用户都是不可见的，因此用户所扩展和继承的solver不能与这些solver类同名  
4.这两部各个solver的具体功能可参看vision_graph/solvers/intuitive_solvers和vision_graph/solvers/function_solvers，这两个目录下由各自的介绍  