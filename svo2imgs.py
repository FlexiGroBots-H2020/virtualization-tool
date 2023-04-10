########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum
import os


class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3


def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():
    
    if not sys.argv or len(sys.argv) != 4:
        sys.stdout.write("Usage: \n\n")
        sys.stdout.write("    ZED_SVO_Export A B C \n\n")
        sys.stdout.write("Please use the following parameters from the command line:\n")
        sys.stdout.write(" A - SVO file path (input) : \"path/to/file.svo\"\n")
        sys.stdout.write(" B - AVI file path (output) or image sequence folder(output) :\n")
        sys.stdout.write("         \"path/to/output/file.avi\" or \"path/to/output/folder\"\n")
        sys.stdout.write(" C - Export mode:  0=Export LEFT+RIGHT AVI.\n")
        sys.stdout.write("                   1=Export LEFT+DEPTH_VIEW AVI.\n")
        sys.stdout.write("                   2=Export LEFT+RIGHT image sequence.\n")
        sys.stdout.write("                   3=Export LEFT+DEPTH_VIEW image sequence.\n")
        sys.stdout.write("                   4=Export LEFT+DEPTH_16Bit image sequence.\n")
        sys.stdout.write(" A and B need to end with '/' or '\\'\n\n")
        sys.stdout.write("Examples: \n")
        sys.stdout.write("  (AVI LEFT+RIGHT):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 0\n")
        sys.stdout.write("  (AVI LEFT+DEPTH):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/file.avi\" 1\n")
        sys.stdout.write("  (SEQUENCE LEFT+RIGHT):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 2\n")
        sys.stdout.write("  (SEQUENCE LEFT+DEPTH):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\" 3\n")
        sys.stdout.write("  (SEQUENCE LEFT+DEPTH_16Bit):  ZED_SVO_Export \"path/to/file.svo\" \"path/to/output/folder\""
                         " 4\n")
        exit()

    # Get input parameters
    svo_input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_as_video = True    
    app_type = AppType.LEFT_AND_RIGHT
    if sys.argv[3] == "1" or sys.argv[3] == "3":
        app_type = AppType.LEFT_AND_DEPTH
    if sys.argv[3] == "4":
        app_type = AppType.LEFT_AND_DEPTH_16
    
    # Check if exporting to AVI or SEQUENCE
    if sys.argv[3] != "0" and sys.argv[3] != "1":
        output_as_video = False

    if not output_as_video and not output_path.is_dir():
        sys.stdout.write("Input directory doesn't exist. Check permissions or create it.\n",
                         output_path, "\n")
        exit()

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2
    
    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    
    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

        
            # Generate file names
            filename1 = output_path / ("%s.png" % str(svo_position).zfill(4))

            # Save Left images
            cv2.imwrite(str(filename1), left_image.get_data())


            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    zed.close()
    return 0


if __name__ == "__main__":
    main()
