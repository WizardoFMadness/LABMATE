import sys

import cv2
import numpy as np


MAX_CAMERA_INDEX = 10
BACKENDS = (
    ("Media Foundation", cv2.CAP_MSMF),
    ("Automatic backend selection", cv2.CAP_ANY),
    ("DirectShow", cv2.CAP_DSHOW),
)


def report_directshow_devices():
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("PyGrabber is unavailable; DirectShow device names cannot be listed.")
        return

    try:
        devices = FilterGraph().get_input_devices()
    except Exception as exc:
        print(f"PyGrabber could not enumerate DirectShow devices: {exc}")
        return

    if devices:
        print("DirectShow device names (informational; these are not OpenCV indices):")
        for index, device in enumerate(devices):
            print(f"  {index}: {device}")
    else:
        print("PyGrabber found no DirectShow devices.")


def describe_fourcc(value):
    value = int(value)
    if not value:
        return "unknown"
    return "".join(chr((value >> (8 * byte)) & 0xFF) for byte in range(4))


def report_frame(frame, cap):
    print(
        "Frame diagnostics: "
        f"shape={frame.shape}, dtype={frame.dtype}, "
        f"channels={1 if frame.ndim == 2 else frame.shape[2]}, "
        f"range={frame.min()}..{frame.max()}, "
        f"reported FOURCC={describe_fourcc(cap.get(cv2.CAP_PROP_FOURCC))}."
    )
    if frame.ndim == 3 and frame.shape[2] == 3:
        means = frame.mean(axis=(0, 1))
        deviations = frame.std(axis=(0, 1))
        green_dominant = np.mean(
            (frame[:, :, 1] > 100)
            & (frame[:, :, 1] > frame[:, :, 0] * 1.35)
            & (frame[:, :, 1] > frame[:, :, 2] * 1.35)
        )
        print(
            "  BGR channel means="
            f"{means.round(1)}, standard deviations={deviations.round(1)}, "
            f"green-dominant pixels={green_dominant:.1%}."
        )
        if green_dominant > 0.8:
            print(
                "  Warning: most pixels are strongly green-dominant. "
                "Compare the raw and RGB-interpreted previews; this may be "
                "a source/driver color-format issue."
            )
    elif frame.ndim not in (2, 3):
        print("  Warning: unexpected frame dimensionality; color preview may fail.")


def find_camera():
    print(f"OpenCV version: {cv2.__version__}")
    report_directshow_devices()
    print(f"Probing OpenCV indices 0-{MAX_CAMERA_INDEX - 1} for a readable frame...")

    for backend_name, backend in BACKENDS:
        print(f"\nTrying {backend_name}:")
        for index in range(MAX_CAMERA_INDEX):
            cap = None
            try:
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    print(f"  Index {index}: could not open")
                    cap.release()
                    continue

                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    print(
                        f"  Connected at index {index} using {backend_name}; "
                        f"frame size is {frame.shape[1]}x{frame.shape[0]}."
                    )
                    report_frame(frame, cap)
                    return cap, frame

                print(f"  Index {index}: opened, but no usable frame was returned")
            except cv2.error as exc:
                print(f"  Index {index}: OpenCV error: {exc}")

            if cap is not None:
                cap.release()

    print(
        "\nNo camera returned a frame. Confirm DroidCam is running, then try "
        "another backend or check whether another application has exclusive access."
    )
    return None, None


def main():
    cap, frame = find_camera()
    if cap is None:
        return 1

    print(
        "Showing the OpenCV frame as received and a diagnostic RGB-interpreted "
        "version. OpenCV normally supplies BGR frames. Press 'q' to quit."
    )
    try:
        while True:
            cv2.imshow("LABMATE - raw OpenCV frame", frame)
            if frame.ndim == 3 and frame.shape[2] == 3:
                interpreted = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("LABMATE - RGB interpretation (diagnostic)", interpreted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ok, frame = cap.read()
            if not ok or frame is None or not frame.size:
                print("Camera stopped returning frames; closing the capture.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
