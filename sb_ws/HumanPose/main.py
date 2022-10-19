def init():
    print("Init")
    pass


def detect_human():
    print("Detection Human: BBox")
    pass


def estimate_pose():
    print("Estimate Human: Keypoints ")
    pass


def release():
    print("Release")
    pass


def main():
    # Init
    init()
    # Huamn Detection
    detect_human()

    # Pose Keypoint Estimation
    estimate_pose()
    # Release
    release()
    pass


if __name__ == "__main__":
    main()
