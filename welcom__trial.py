import cv2  # For webcam access and image display
from gtts import gTTS, gTTSError  # For text-to-speech
import os  # For file operations
import time  # For delays
import numpy  # For creating blank images
import face_recognition  # For face detection and recognition
import pygame.mixer  # For audio playback
import threading  # For running audio in the background
import requests  # For network connectivity checks
import imageio  # For reading GIF files
# import screeninfo  # Optional: for dynamic screen resolution (uncomment if installed)

# Define known faces (replace these with your own image paths)
known_faces = {
    "nandini": "nandini.jpg",
    "aarushi": "aarushi.jpg",
    "tripti": "tripti.jpg",
    "jagdishsar": "jagdishsar.jpg",
    "KamleshSar": "KamlesSar.jpg"
}

# Load known faces into encodings
def load_known_faces(known_faces):
    """
    Load face images and generate their encodings.
    Returns a dictionary mapping names to face encodings.
    """
    encodings = {}
    for name, image_path in known_faces.items():
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            encodings[name] = encoding
        except FileNotFoundError:
            print(f"Error: {image_path} not found")
        except IndexError:
            print(f"Error: No face found in {image_path}")
    return encodings

# Initialize known face encodings and names
known_face_encodings = load_known_faces(known_faces)
known_face_names = list(known_face_encodings.keys())

# Recognize face using webcam
def recognize_face():
    """
    Capture video from webcam and recognize faces.
    Returns the name of the recognized person or None if no match.
    """
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                list(known_face_encodings.values()), 
                face_encoding,
                tolerance=0.6
            )
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw rectangle and label on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Return the name (including "Unknown" for processing)
            video_capture.release()
            cv2.destroyAllWindows()
            return name

        # Display the webcam feed
        cv2.imshow('Webcam - Look at the camera!', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q'
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return None

# Generate and play audio with network check
def generate_and_play_audio(name):
    """
    Generate a personalized welcome message and play it.
    Includes network check to avoid gTTS connection errors.
    """
    print(f"Starting audio process for {name}")
    
    # Check network connectivity
    try:
        response = requests.get('https://www.google.com', timeout=10)
        if response.status_code != 200:
            print("Network error: Unable to reach Google.")
            return
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return
    
    # Delay to avoid rate limiting by Google
    time.sleep(2)
    
    # Custom welcome messages
    if name == "KamleshSar":
        welcome_message = (
            "It is my great honor and privilege to welcome our esteemed President, Mr. Kamlesh Mishra, "
            "to this special demonstration of our Center of Excellence powered by AI."
        )
    elif name == "jagdishsar":
        welcome_message = (
            "Esteemed Mr. Jagdish Sir, it’s a privilege to welcome you, our dynamic Vice President, to this special demonstration of our Center of Excellence (COE) powered by AI."
        )
    elif name == "Unknown":
        welcome_message = (
            "Dear guest, we warmly welcome you to this special demonstration of our Center of Excellence "
            " powered by AI. We’re delighted to have you join us today!"
        )
    else:
        welcome_message = (
            f"Greetings, {name}! We are delighted to welcome you to this special demonstration of our "
            "Center of Excellence powered by AI, where innovation meets excellence."
        )
    
    print(f"Generating audio: '{welcome_message}'")
    audio_file = f"welcome_{name.replace(' ', '_')}.mp3"
    
    try:
        # Generate audio using gTTS
        tts = gTTS(text=welcome_message, lang='en', slow=False)
        tts.save(audio_file)
        print(f"Audio file saved: {audio_file}")
        
        if os.path.exists(audio_file):
            print(f"File exists: {audio_file}")
            pygame.mixer.init()
            pygame.mixer.music.load(audio_file)
            print(f"Loaded audio into pygame: {audio_file}")
            pygame.mixer.music.play()
            print("Playing audio...")
            while pygame.mixer.music.get_busy():  # Wait for audio to finish
                time.sleep(0.1)
            pygame.mixer.quit()
            print("Audio playback completed")
            os.remove(audio_file)  # Clean up audio file
            print(f"Removed: {audio_file}")
        else:
            print(f"Error: Audio file {audio_file} not created")
    
    except gTTSError as e:
        print(f"gTTS error: {e}")
    except Exception as e:
        print(f"Other error: {type(e).__name__} - {str(e)}")

# Main function to orchestrate the system with full-screen GIF support
def main():
    """
    Main loop to recognize faces, display welcome GIF in full screen, and play audio simultaneously.
    """
    print("Starting trial... Look at the webcam!")
    while True:
        name = recognize_face()
        if name:  # Process all recognized names, including "Unknown"
            print(f"Recognized: {name}")
            # Load the GIF using imageio
            try:
                gif = imageio.mimread("avatar.gif")  # Replace with your GIF file path
                gif_frames = [cv2.cvtColor(numpy.array(frame), cv2.COLOR_RGB2BGR) for frame in gif]
                print("GIF loaded successfully")
            except FileNotFoundError:
                print("Error: avatar.gif not found. Showing text instead.")
                blank_image = numpy.zeros((300, 500, 3), dtype="uint8")
                cv2.putText(blank_image, f"Welcome {name}!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                gif_frames = [blank_image]  # Fallback to a single frame

            # Get screen resolution (default or dynamic)
            # Uncomment the following if using screeninfo:
            # screen = screeninfo.get_monitors()[0]  # Get primary monitor
            # screen_width, screen_height = screen.width, screen.height
            screen_width, screen_height = 1920, 1080  # Default resolution

            # Resize GIF frames to full screen
            resized_gif_frames = [cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_AREA) for frame in gif_frames]

            # Start audio playback in a separate thread
            audio_thread = threading.Thread(target=generate_and_play_audio, args=(name,))
            audio_thread.start()

            # Set up full-screen window
            cv2.namedWindow("Welcome Window", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Welcome Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Display GIF frames in a loop while audio plays
            frame_index = 0
            while audio_thread.is_alive():
                cv2.imshow("Welcome Window", resized_gif_frames[frame_index])
                frame_index = (frame_index + 1) % len(resized_gif_frames)  # Loop through frames
                if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay per frame, adjustable
                    break

            # Close window after audio finishes
            cv2.destroyAllWindows()
            time.sleep(3)  # Pause before next recognition
        else:
            print("No face recognized")
            time.sleep(1)  # Brief pause before retrying

if __name__ == "__main__":
    main()
