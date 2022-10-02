#ifndef _BOX_H_
#define _BOX_H_

#include <cassert>
#include "opencv2/opencv.hpp"


namespace vision {

struct Box
{
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float score = 0.0;
    std::vector<int> ahash;

    int cls = 0; // class id
    int tid = -1;
    long update_time = 0;
    int move_state = 0;

    void clear()
    {
        x1 = x2 = y1 = y2 = score = 0.0;
        cls = 0;
    }

    bool has_rect = false;
    cv::Rect rect;
    cv::Rect box2rect(Box &b) const
    {
        cv::Rect rect;
        rect.x = b.x1, rect.y = b.y1;
        rect.width = b.width(), rect.height = b.height();
        return rect;
    }

    Box rect2box(cv::Rect &rect) const
    {
        Box b;
        b.x1 = rect.x, b.y1 = rect.y;
        b.x2 = b.x1 + rect.width, b.y2 = b.y1 + rect.height;
        return b;
    }

    float dist(const Box &b) const
    {
        float center_x = (x1 + x2) / 2;
        float center_y = (y1 + y2) / 2;

        float d_x = (b.x1 + b.x2) / 2;
        float d_y = (b.y1 + b.y2) / 2;

        float ret = std::abs(center_x - d_x) + std::abs(center_y - d_y);
        return ret;
    }

    bool empty() const
    {
        if (std::fabs(x2 - x1) <= 1 && std::fabs(y2 - y1) <= 1)
        {
            return true;
        }
        return false;
    }

    float area() const
    {
        return std::fabs(x2 - x1) * std::fabs(y2 - y1);
    }

    void add(const Box &b)
    {
        x1 += b.x1;
        y1 += b.y1;
        x2 += b.x2;
        y2 += b.y2;
    }

    void divide(uint32_t base)
    {
        x1 /= base;
        y1 /= base;
        y2 /= base;
        x2 /= base;
    }

    void scale(float ratio)
    {
        x1 *= ratio;
        y1 *= ratio;
        x2 *= ratio;
        y2 *= ratio;
    }

    /**
     * [min, max)
     * @param val
     * @param min
     * @param max
     * @return
     */
    float checkInRange(float val, int min, int max) const
    {
        if (val < min)
        {
            val = min;
        }
        if (val >= max)
        {
            val = max - 1;
        }
        return val;
    }

    /**
     * 缩放到边长为side，若越界，以不越界为标准重新缩放。
     * @param side
     * @param maxw
     * @param maxh
     * @return
     */
    float expandLongMaxToPixel(float side, int maxw, int maxh)
    {
        float centerx = (x1 + x2) / 2;
        float centery = (y1 + y2) / 2;
        float w = x2 - x1;
        float h = y2 - y1;
        long update_time;
        float _long = std::max(w, h);
        if (_long < side)
        {
            return 1.0f;
        }

        float ratio = side / _long;

        _long = side;
        x1 = centerx - _long / 2;
        x2 = centerx + _long / 2;
        y1 = centery - _long / 2;
        y2 = centery + _long / 2;

        //        x1 = checkInRange(x1, 0, maxw);
        //        y1 = checkInRange(y1, 0, maxh);
        //        x2 = checkInRange(x2, 0, maxw);
        //        y2 = checkInRange(y2, 0, maxh);
        return ratio;
    }

    /**
     * 基于原尺寸扩大一部分，保证不越界
     * @param ratio
     * @param maxw
     * @param maxh
     */
    void expandByLongSide_face(float ratio, cv::Point p)
    {
        float centerx = p.x;
        float centery = p.y;
        //        float centerx = (x1 + x2) / 2;
        //        float centery = (y1 + y2) / 2;
        float w = x2 - x1;
        float h = y2 - y1;
        float _long = w > h ? w : h;

        _long *= ratio;
        x1 = centerx - _long / 2;
        x2 = centerx + _long / 2;
        y1 = centery - _long / 2;
        y2 = centery + _long / 2;

        //        x1 = checkInRange(x1, 0, maxw);
        //        y1 = checkInRange(y1, 0, maxh);
        //        x2 = checkInRange(x2, 0, maxw);
        //        y2 = checkInRange(y2, 0, maxh);
    }
    void expandByLongSide(float ratio, int maxw, int maxh)
    {
        float centerx = (x1 + x2) / 2;
        float centery = (y1 + y2) / 2;
        float w = x2 - x1;
        float h = y2 - y1;
        float _long = w > h ? w : h;

        _long *= ratio;
        x1 = centerx - _long / 2;
        x2 = centerx + _long / 2;
        y1 = centery - _long / 2;
        y2 = centery + _long / 2;

        x1 = checkInRange(x1, 0, maxw);
        y1 = checkInRange(y1, 0, maxh);
        x2 = checkInRange(x2, 0, maxw);
        y2 = checkInRange(y2, 0, maxh);
    }

    void anti_flip(int flipCode, int w, int h)
    {
        if (0 == flipCode)
        {
            float tmp_y1 = h - y2;
            float tmp_y2 = h - y1;
            y1 = tmp_y1;
            y2 = tmp_y2;
        }
        else if (1 == flipCode)
        {
            float tmp_x1 = w - x2;
            float tmp_x2 = w - x1;
            x1 = tmp_x1;
            x2 = tmp_x2;
        }
    }

    /**
     * transpose后的坐标改回去
     * @param oldw
     * @param oldh
     */
    void anti_transpose()
    {
        float tmp = x1;
        x1 = y1;
        y1 = tmp;

        tmp = x2;
        x2 = y2;
        y2 = tmp;
    }

    inline float width() const
    {
        return x2 - x1;
    }

    inline float height() const
    {
        return y2 - y1;
    }
};

} // namespace vision
#endif
