import cv2

def extract_frames(video_path, fps_target=24):

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file opened sucessfully
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return[]

    # Get the frame rate of the video
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames to extract
    frame_interval = int(fps / fps_target)
    
    frame_count = 0
    extracted_frames = []

    while True:
        # Read the next frame from the video
        success, frame = video_capture.read()

        # Break the loop if no more frames are available
        if not success:
            break

        # Extract one frame every 'frame_interval' frames
        if frame_count % frame_interval == 0:
            extracted_frames.append(frame)

        frame_count += 1

    # Release the video capture object
    video_capture.release()

    return extracted_frames
