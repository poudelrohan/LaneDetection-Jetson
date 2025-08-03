import pyzed.sl as sl

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Create initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200  # Adjust if needed
    init_params.camera_fps = 30  # Set FPS (you can change)

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(1)

    # Prepare runtime parameters
    runtime_params = sl.RuntimeParameters()

    # Create an image container for the frames
    image = sl.Mat()

    print("Start capturing 50 frames...")

    for i in range(50):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            print(f"Captured frame {i+1}")

            # Optionally, access the image data here for further processing
            # For example, convert to numpy array if needed:
            # import numpy as np
            # frame_np = image.get_data()

        else:
            print(f"Frame {i+1} grab failed.")

    # Close the camera
    zed.close()
    print("Done capturing.")

if __name__ == "__main__":
    main()
