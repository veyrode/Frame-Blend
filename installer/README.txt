Frame Blend 1.4.2
=================

Frame Blend turns high frame rate recordings (240, 480, 600, 1000+ fps) into
smooth lower frame rate video (for example 30 or 60 fps) by blending every
recorded frame that each output frame covers, like VEGAS Pro's frame blending.

Requirements: Windows 10 or 11 (64-bit), Adobe Premiere Pro or After Effects
2020 or later. Premiere renders Frame Blend on the GPU (NVIDIA CUDA or OpenCL).


Premiere Pro
------------
- Put a high frame rate clip on a sequence with a lower frame rate. Frame
  Blend is added to it automatically, with the clip's Time Interpolation set
  to Frame Blending.
- To turn it off for a clip: right-click the clip > Time Interpolation >
  Frame Sampling (Frame Blending turns it back on). This works on several
  selected clips at once.
- You can also add it by hand: Effects panel > Video Effects > Frame Blending
  > Frame Blend.

After Effects
-------------
- Effect > Frame Blending > Frame Blend on a high frame rate layer in a lower
  frame rate composition. Works in 8, 16 and 32 bits per channel projects.
- The layer's frame rate comes from the footage interpretation (Interpret
  Footage > Frame Rate).


Troubleshooting
---------------
To record a log for a bug report, create an empty file named
FrameBlend-log.txt in your temp folder (type %TEMP% in the Explorer address
bar), restart Premiere or After Effects, reproduce the problem, and send that
file. Delete it afterwards: the log is only written while it exists.


Uninstall
---------
Settings > Apps > Installed apps > Frame Blend > Uninstall.


Third-party software
--------------------
Frame Blend uses FFmpeg (https://ffmpeg.org), licensed under the GNU LGPL
version 3 (FFmpeg-LICENSE.txt, next to this file). The FFmpeg libraries are
the unmodified shared builds avutil-60.dll, swresample-6.dll, avcodec-62.dll
and avformat-62.dll from https://github.com/BtbN/FFmpeg-Builds (release
n8.1, LGPL shared), installed beside FrameBlend.aex; they can be replaced by
other builds of the same version.
