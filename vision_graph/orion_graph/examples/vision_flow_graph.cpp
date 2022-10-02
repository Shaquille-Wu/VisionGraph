#include "tbb/flow_graph.h"
#include <iostream>

#include <cstring>
#include <cstdio>

const int   kImgWidth   = 640;
const int   kImgHeight  = 480;
const int   kImgChannel = 3;
const int   kImgSize    = kImgWidth * kImgHeight * kImgChannel;

struct OImage 
{
    OImage();

    OImage(const OImage& other):image_size_(0),data_(nullptr)
    {
        if(other.image_size_ > 0)
        {
            image_size_ = other.image_size_;
            data_       = new unsigned char[image_size_];
        }
    }

    OImage& operator=(const OImage& other)
    {
        if(&other == this)
            return *this;
        
        if(nullptr != data_)
        {
            delete data_;
        }
            
        data_       = nullptr;
        image_size_ = 0;
        if(other.image_size_ > 0)
        {
            image_size_ = other.image_size_;
            data_       = new unsigned char[image_size_];
        }

        return *this;
    }

    OImage(int image_count, bool a, bool b );
    
    virtual ~OImage()
    {
        if(nullptr != data_)
            delete data_;
        data_ = nullptr;
    }

    int             image_size_;
    unsigned char  *data_;
};

OImage::OImage() : image_size_(kImgSize) 
{
   data_ = new unsigned char[image_size_];
}

OImage::OImage(int image_count, bool a, bool b ) : image_size_(kImgSize) 
{
    data_ = new unsigned char[image_size_];
    memset(data_, 0, image_size_);
    data_[0] = ((unsigned char)image_count) - 32;
    if ( a ) data_[image_size_-2] = 'A';
    if ( b ) data_[image_size_-1] = 'B';
}

int        cur_image_count    = 0;
int        image_count        = 64;
const int  kFreqA             = 11;
const int  kFreqB             = 13;

OImage *get_next_image() 
{
    bool a = false, b = false;
    if ( cur_image_count < image_count ) 
    {
        if ( cur_image_count%kFreqA == 0 ) a = true;
        if ( cur_image_count%kFreqB == 0 ) b = true;
        return new OImage( cur_image_count++, a, b );
    } 
    else 
       return nullptr;
}

void preprocess_image(OImage *input_image, OImage *output_image ) 
{
    for ( int i = 0; i < input_image->image_size_; ++i ) 
        output_image->data_[i] = input_image->data_[i] + 32;
}

bool detect_with_A(OImage *input_image) 
{
    for (int i = 0; i < input_image->image_size_; ++i) 
    {
        if ( input_image->data_[i] == 'a' )
            return true;
    }
    return false;
}

bool detect_with_B(OImage *input_image ) 
{
    for (int i = 0; i < input_image->image_size_; ++i) 
    {
        if (input_image->data_[i] == 'b')
            return true;
    }
    return false;
}

void output_image(OImage *input_image, bool found_a, bool found_b ) 
{
    bool a = false, b = false;
    int a_i = -1, b_i = -1;
    for ( int i = 0; i < input_image->image_size_; ++i ) 
    {
        if ( input_image->data_[i] == 'a') 
        { 
            a   = true; 
            a_i = i; 
        }
        if ( input_image->data_[i] == 'b') 
        { 
            b    = true; 
            b_i  = i; 
        }
    }
    printf("Detected feature (a,b)=(%d,%d)=(%d,%d) at (%d,%d) for image %p:%d\n",
            a, b, found_a, found_b, a_i, b_i, input_image, input_image->data_[0]) ;
}


const int kGraphBuffers = 8;

bool src_continue = false;

int main() 
{
    int                                                   i = 0;
    tbb::flow::graph                                      g;
    typedef std::tuple<OImage*, OImage*>                  resource_tuple;
    typedef std::pair<OImage*, bool>                      detection_pair;
    typedef std::tuple< detection_pair, detection_pair>   detection_tuple;

    tbb::flow::queue_node<OImage*>                                  buf_que_node(g);
    tbb::flow::join_node<resource_tuple, tbb::flow::queueing>       resource_join(g);
    tbb::flow::join_node<detection_tuple, tbb::flow::tag_matching>  detection_join(g, 
                                                                                   [](const detection_pair &p) -> size_t { 
                                                                                       return (size_t)p.first; 
                                                                                   },
                                                                                   [](const detection_pair &p) -> size_t { 
                                                                                       return (size_t)p.first; 
                                                                                   });
/*                                                                     
    tbb::flow::source_node<OImage*> src( g,
                                        []( OImage* &next_image ) -> bool {
                                            next_image = get_next_image();
                                            if ( next_image ) 
                                               return true;
                                            else 
                                               return false;
                                          });
*/

    tbb::flow::function_node<OImage*, OImage*> src( g, tbb::flow::unlimited,
                                        []( OImage* const& next_image ) -> OImage* {
                                            return get_next_image();
                                          });

    //tbb::flow::broadcast_node<OImage*>  src(g);
    tbb::flow::make_edge(src,            tbb::flow::input_port<0>(resource_join));
    tbb::flow::make_edge(buf_que_node,   tbb::flow::input_port<1>(resource_join));

    tbb::flow::function_node<resource_tuple, OImage*>    preprocess_function( g, tbb::flow::unlimited,
                                                                             [](const resource_tuple &in ) -> OImage* {
                                                                                    OImage* input_image  = std::get<0>(in);
                                                                                    OImage* output_image = std::get<1>(in);
                                                                                    preprocess_image(input_image, output_image);
                                                                                    delete input_image;
                                                                                    return output_image;
                                                                                });
    tbb::flow::make_edge(resource_join, preprocess_function);

    tbb::flow::function_node<OImage*, detection_pair>    detect_A(g, tbb::flow::unlimited, 
                                                                    [](OImage* input_image) -> detection_pair {
                                                                        bool r = detect_with_A( input_image);
                                                                        return std::make_pair(input_image, r);
                                                                    });

    tbb::flow::function_node<OImage*, detection_pair>    detect_B(g, tbb::flow::unlimited,
                                                                    [](OImage *input_image) -> detection_pair {
                                                                        bool r = detect_with_B( input_image);
                                                                        return std::make_pair(input_image, r);
                                                                    });
    tbb::flow::make_edge(preprocess_function, detect_A);
    tbb::flow::make_edge(detect_A, tbb::flow::input_port<0>(detection_join));
    tbb::flow::make_edge(preprocess_function, detect_B);
    tbb::flow::make_edge(detect_B, tbb::flow::input_port<1>(detection_join));

    tbb::flow::function_node<detection_tuple, OImage*>  decide(g, tbb::flow::serial,
                                                                []( const detection_tuple &t) -> OImage* {
                                                                    const detection_pair &a = std::get<0>(t);
                                                                    const detection_pair &b = std::get<1>(t);
                                                                    OImage* img = a.first;
                                                                    if ( a.second || b.second ) 
                                                                    {
                                                                        output_image( img, a.second, b.second );
                                                                    }
                                                                    return img;
                                                                });
    
    tbb::flow::make_edge(detection_join, decide);
    tbb::flow::make_edge(decide,         buf_que_node);

    
    //g.set_active(false);
    // Put image buffers into the buffer queue
    for (i = 0; i < kGraphBuffers; ++i) 
    {
        OImage *img = new OImage;
        buf_que_node.try_put(img);
        src.try_put(new OImage);
    }

    //g.set_active(true);
    g.wait_for_all();
    std::cout << "graph idle" << std::endl;

    for (i = 0; i < kGraphBuffers; ++i) 
    {
        OImage *img = nullptr;
        if ( !buf_que_node.try_get(img) )
            printf("ERROR: lost a buffer\n");
        else
            delete img;
    }
    return 0;
}