## draw_graph
基本功能：依据graph的json文件，画出该json文件中所有子图的数据流图，保存成png格式  
环境依赖：程序依赖Graphviz，因此需要运行环境安装了Graphviz  
命令格式：
```
./draw_graph -j json_file
```
假设有一个json文件叫vision_graph.json，那么命令如下：
```
./draw_graph -j vision_graph.json
```
假设vision_graph.json里面有2个子图分别叫sub_graph0和sub_graph1，那么在vision_graph.json的同级目录下将生成2个文件：vision_graph_sub_graph0.png和vision_graph_sub_graph1.png  