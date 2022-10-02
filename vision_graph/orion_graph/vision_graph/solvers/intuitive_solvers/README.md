## SolverData
如果一个node的solver被定义成了SolverData，那么这个结点就是输入结点，其input的数量为0  
### 输入
无，input的数量必须为0  
### 输出
数量必须为1，目前的类型包括:   
``` 
    { "UInt8",             kTensorUInt8             },
    { "Int8",              kTensorInt8              },
    { "UInt16",            kTensorUInt16            },
    { "Int16",             kTensorInt16             },
    { "UInt32",            kTensorUInt32            },
    { "Int32",             kTensorInt32             },
    { "UInt64",            kTensorUInt64            },
    { "Int64",             kTensorInt64             },
    { "Float32",           kTensorFloat32           },
    { "Float64",           kTensorFloat64           },
    { "String",            kTensorString            },
    { "Point",             kTensorPoint             },
    { "Box",               kTensorBox               },
    { "Image",             kTensorImage             },
    { "BoxesMap",          kTensorBoxesMap          },
    { "KeyPoints",         kTensorKeyPoints         },
    { "Attributes",        kTensorAttributes        },
    { "Feature",           kTensorFeature           },
    { "FeatureMaps",       kTensorFeatureMaps       },
    { "Reference",         kTensorReference         },
    { "UInt8Vector",       kTensorUInt8Vector       },
    { "Int8Vector",        kTensorInt8Vector        },
    { "UInt16Vector",      kTensorUInt16Vector      },
    { "Int16Vector",       kTensorInt16Vector       },
    { "UInt32Vector",      kTensorUInt32Vector      },
    { "Int32Vector",       kTensorInt32Vector       },
    { "UInt64Vector",      kTensorUInt64Vector      },
    { "Int64Vector",       kTensorInt64Vector       },
    { "Float32Vector",     kTensorFloat32Vector     },
    { "Float64Vector",     kTensorFloat64Vector     },
    { "BoxVector",         kTensorBoxVector         },
    { "ImageVector",       kTensorImageVector       },
    { "KeypointsVector",   kTensorKeypointsVector   },
    { "PtrVector",         kTensorPtrVector         }
``` 
### json配置格式
``` 
"solver":  {
	"name":      "image",
	"class":     "vision_graph::SolverData",
	"data_type": "Image"    ##指定了要输出的数据类型
}
``` 

## SolverCounter
执行计数操作，计数方式可以是"increase"和"decrease"， 同时可以设置计数周期（cycle）  
### 输入
无，input的数量必须为0，只能且必须通过enable_ports的方式对其进行触发  
### 输出
数量必须为1，输出类型为TensorInt64  
### json配置格式
``` 
"solver":  {
	"name":         "counter",
	"class":        "vision_graph::SolverCounter",
	"counter_type": "increase",        ##填写"increase"或者"decrease"
	"cycle":        30                 ##填写计数周期,达到这个周期即从0开始
	                                   ##如果json中没有"cycle"，或者"cycle"填0，则一直正常往下计数，不考虑周期
}
``` 

## SolverLogic
执行与、或、非和取反等逻辑操作，一共四种类型：  
	{ "and",  SolverLogic::AND },  
    { "or",   SolverLogic::OR  },  
    { "xor",  SolverLogic::XOR },  
    { "not",  SolverLogic::NOT }  
每一种逻辑操作都支持两种方式：0/1操作(操作符两端的左值和右值都会被转成0和非0，非0即是1)和按位操作，在json中标明"bitwise"即表示按位操作，否则按0/1操作  
无论哪种方式，其输出值都是一个UInt32  
### 输入
双目操作时，其输入必须是两个数，且都是一个标量，包括所有被TensorNumeric定义的数值型数据类型。双目操作包括： and, or, xor  
单目操作时，其输入必须是一个数，且必须是一个标量，包括所有被TensorNumeric定义的数值型数据类型。单目操作包括： not  
因为其输出是一个UInt32，如果输入参数是float32或者float64，那么，会将这个float32或者float64进行舍入操作，再转成UInt32进行逻辑运算  
### 输出
数量必须为1，类型为TensorUInt32  
### json配置格式
``` 
"solver":  {
	"name":       "logic_compare",
	"class":      "vision_graph::SolverLogic",
	"logic_type": ##"logic_type"##
	"bitwise":    ##true_or_false##
}
``` 
logic_type, 4种选择，"and", "or", "xor", "not"  
true_or_false, 2种选择，true, false  

## SolverCompare
执行比较操作，一共6种类型：  
    { ">=",  SCALAR_COMPARE_TYPE::GTE },  
    { ">",   SCALAR_COMPARE_TYPE::GT },  
    { "<=",  SCALAR_COMPARE_TYPE::LTE },  
    { "<",   SCALAR_COMPARE_TYPE::LT },  
    { "==",  SCALAR_COMPARE_TYPE::EQ },  
    { "!=",  SCALAR_COMPARE_TYPE::NEQ }  
json中的配置包括： 
``` 
	"compare_type": "##compare_type##",
	"right_value":{
		"source": "##source_type##",
		"value":  ##source_value##
	}
```
compare_type，包括上述6种  
source_type，包括"var"和"const"，"var"表示右值是变量，"const"表示右值是常量  
source_value，如果右值的类型是"const"，则必须填写这个参数，  
              1）填写整数，则这个右值是Int32  
              2）填写浮点数，则这个右值是Float32  
              3）填写字符串，则这个右值是字符串  

### 输入
当右值是var时，其输入必须是两个数，支持标量（包括所有被TensorNumeric定义的数值型数据类型）和字符串，如果两个变量中一个是标量另一个是字符串，这属于非法操作，程序退出   
当右值是const时，其输入必须是一个数，支持标量（包括所有被TensorNumeric定义的数值型数据类型）和字符串。如果右值是标量，那么输入也必须是标量，如果右值是字符串，那么输入也必须是字符串，否则属于非法操作，程序退出  
### 输出
数量必须为1，类型为UInt32，结果只能是0或1    
### json配置格式
``` 
	"solver":  {
		"name":         "frame_compare",
		"class":        "vision_graph::SolverCompare",
	    "compare_type": "##compare_type##",
	    "right_value":{
		    "source": "##source_type##",
		    "value":  ##source_value##
	    }
	}
```
具体介绍如上所述   			  

## SolverBranch
通过输入的数字来判断后面应该走哪个分支  
支持4种"forward_type":  
    { "var_num",    SolverBranch::VAR_NUM }  
    { "var_mask",   SolverBranch::VAR_MASK }  
    { "fixed_num",  SolverBranch::FIXED_NUM }  
    { "fixed_mask", SolverBranch::FIXED_MASK }  
var_num:   根据输入的值，程序跳往其指定的分支。输入数据是一个数字，支持所有被TensorNumeric定义的数值型数据类型，如果是浮点型数据，将对其进行舍入操作，最终参与运算的是UInt32，输出结果也是一个UInt32，因此输入的有符号数不能是一个负数  
var_mask:  类似0111000111，这样的二进制串，程序根据输入跳往被标识成1的分支  
fixed_num:  与var_num类似，只是程序不会根据输入数据进行跳转，只会根据json种的配置进行跳转，某些调试状态下才会使用    
fixed_mask: 与var_mask类似，只是程序不会根据输入数据进行跳转，只会根据json种的配置进行跳转，某些调试状态下才会使用    
### 输入
输入的数量必须是1，且输入的数据类型必须是一个标量（所有被TensorNumeric定义的数值型数据类型），如果是浮点型数据，将对其进行舍入操作，转成UInt32  
### 输出
输出的数量不限，其类型被定义成UInt32，往哪个分支跳转，这个分支上的输出被写为1，否则为0  
### json配置格式
``` 
"solver":  {
	"name":         "counter_branch",
	"class":        "vision_graph::SolverBranch",
	"forward_type": ##"forward_type"##            ##上面介绍的4种选择
	"value":        ##"fixed_forward_value"##     ##如果forward_type是fixed_***，则需要填写该参数，否则无需填写  
}
```

## SolverSelector
对输入数据进行选择，并往后继节点传送，支持4种select_type：  
    { "var_num",      SolverSelector::VAR_NUM },  
    { "fixed_num",    SolverSelector::FIXED_NUM },  
    { "compare",      SolverSelector::COMPARE },  
    { "pass_through", SolverSelector::PASS_THROUGH },  
var_num，由输入索引为0的数据决定选择哪个输入数据（输入索引>=1），最终输出一个数据，var_num等于0，则输出索引为1的输入数据，等于1，则输出2，依次类推  
fixed_num，由json中配置的常数，决定选择哪个输入数据（输入索引>=0），最终输出一个数据，fixed_num等于0，则输出索引为1的输入数据，等于1，则输出2，依次类推  
compare，由输入索引为0的数据和json中配置的数据比较（包括>, >=, ==, <, <=），经比较，若得到false，则输出输入索引为1的输入数据，若得到true，则输出输入索引为2的输入数据  
pass_through，其输入的数目必须为1，直接通过  
### 输入
如果select_type是var_num，则输入的数量必须是1+x，且第0个输入用于指定输出哪个，其数据类型必须是一个标量（所有被TensorNumeric定义的数值型数据类型），如果是浮点型数据，将对其进行舍入操 作，转成UInt32，其他输入作为待选择数据      
如果select_type是fix_num，则输入的数量必须大于fix_num.value，且所有的输入数据都是待选择数据  
如果select_type是compare，则输入的数量必须是1+x，且第0个输入用于与json中的"value"做比较，其他输入作为待选择数据      
如果select_type是pass_through，则输入的数量必须是1，这个输入数据作为待选择数据  
### 输出
输出的数量必须是1，其类型一定是TensorReference， 通过其reference成员获得其选择的结果    
### json配置格式
``` 
"solver":  {
	"name":         "target_selector",
	"class":        "vision_graph::SolverSelector",
	"select_type":  ##"select_type"##,     #选择"var_num","fixed_num","compare","pass_through"
	"compare_type": ##"compare_type"##,    #如果select_type是compare，则选择">", ">=", "==", "<", "<="，否则不需要填写
	"value"：       ##value##,             #如果select_type是fix_num或者compare，则选择需要填写，整形、浮点或者字符串，需要与输入索引为0的数据类型一致，其中数据值中的各种定点和浮点，程序可以自动强转    
}
```


## to be continue ...