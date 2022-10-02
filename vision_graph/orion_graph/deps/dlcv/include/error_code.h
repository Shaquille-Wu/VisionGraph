#ifndef _ERROR_CODE_H_
#define _ERROR_CODE_H_

namespace vision {

constexpr int ERROR_CODE_SUCCESS        =      0;
constexpr int ERROR_CODE_IMAGE_VALID    =      0;
constexpr int ERROR_CODE_PARAM_NULL     =  -1100;
constexpr int ERROR_CODE_PARAM_SIZE     =  -1101;
constexpr int ERROR_CODE_IMAGE_DECODE   =  -1102;
constexpr int ERROR_CODE_IMAGE_EMPTY    =  -1103;
constexpr int ERROR_CODE_CONFIG_FORMAT  =  -1104;
constexpr int ERROR_CODE_LICENCE_CHECK  =  -1105;
constexpr int ERROR_CODE_OUT_OF_MEMORY  =  -1106;
constexpr int ERROR_CODE_SHAPE_MISMATCH =  -1107;

} // namespace vision

#endif
