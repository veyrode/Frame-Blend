#include "AEConfig.h"
#include "AE_EffectVers.h"

resource 'PiPL' (16000) {
    {   /* array properties: 12 elements */
        /* [1] */
        Kind {
            AEEffect
        },
        /* [2] */
        Name {
            "Frame Blend"
        },
        /* [3] */
        Category {
            "Frame Blending"
        },
#ifdef AE_OS_WIN
    #if defined(AE_PROC_INTELx64)
        CodeWin64X86 {"EffectMain"},
    #elif defined(AE_PROC_ARM64)
        CodeWinARM64 {"EffectMain"},
    #endif
#elif defined(AE_OS_MAC)
        CodeMacIntel64 {"EffectMain"},
        CodeMacARM64 {"EffectMain"},
#endif
        /* [6] */
        AE_PiPL_Version {
            2,
            0
        },
        /* [7] */
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        /* [8] */
        AE_Effect_Version {
            655360 /* PF_VERSION(1, 4, 0, DEVELOP, 0) */
        },
        /* [9] */
        AE_Effect_Info_Flags {
            0
        },
        /* [10] */
        AE_Effect_Global_OutFlags {
            33555462 /* WIDE_TIME_INPUT | NON_PARAM_VARY | PIX_INDEPENDENT | DEEP_COLOR_AWARE */
        },
        AE_Effect_Global_OutFlags_2 {
            134222848 /* FLOAT_COLOR_AWARE | SUPPORTS_SMART_RENDER | SUPPORTS_THREADED_RENDERING */
        },
        /* [11] */
        AE_Effect_Match_Name {
            "FrameBlend Native"
        },
        /* [12] */
        AE_Reserved_Info {
            0
        }
    }
};
