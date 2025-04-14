#ifdef USE_SENTENCEPIECE

/* This should be the only place this warning is suppressed to get Windows build working. 
 * Protobuf dependency creates W4100, which cannot be modified.
 */

#pragma warning(disable : 4100)

#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"

/* https://github.com/google/googletest/issues/1063#issuecomment-332518392 */	
#if __GNUC__ >= 5	
// Disable GCC 5's -Wsuggest-override warnings in gtest	
# pragma GCC diagnostic push	
# pragma GCC diagnostic ignored "-Wsuggest-override"	
#endif	

/* Current inclusion of SentencePiece structures assume builtin-protobuf hard. 
 * Future TODO: Make it work with standard protobuf as well.
 * */
#include "sentencepiece/src/builtin_pb/sentencepiece.pb.h"	

#if __GNUC__ >= 5	
# pragma GCC diagnostic pop	
#endif	

#endif
