#!/usr/bin/env python3
import pyzed.sl as sl
import cv2

def main(num_frames=50):
    # Initialize ZED camera
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200  # 1920x1200, good for ZED X
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # Disable depth for speed

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {zed.get_last_error()}")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    left_image = sl.Mat()

    print(f"Capturing {num_frames} frame(s) from the LEFT camera...")

    for i in range(num_frames):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            frame = left_image.get_data()

            # Optional: Resize frame here if needed

            cv2.imshow("ZED X - Left View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early exit.")
                break
        else:
            print(f"Frame {i+1}: Grab failed.")

    zed.close()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main(num_frames=100)
