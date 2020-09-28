import sys
import getopt
import cv2
import numpy as np
from morphological_operator import binary
from morphological_operator import gray


def operator(
    in_file, out_file, mor_op, wait_key_time=0, input_text_file="", output_text_file=""
):
    img_origin = cv2.imread(in_file)
    cv2.imshow("original image", img_origin)
    cv2.waitKey(wait_key_time)

    img_gray = cv2.imread(in_file, 0)
    cv2.imshow("gray image", img_gray)
    cv2.waitKey(wait_key_time)

    (thresh, img) = cv2.threshold(
        img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    cv2.imshow("binary image", img)
    cv2.waitKey(wait_key_time)

    kernel = np.ones((5, 5), np.uint8)
    kernel_gray = np.zeros((5, 5), np.uint8)
    img_out = None

    if mor_op == "dilate":
        img_dilation = cv2.dilate(img, kernel)
        cv2.imshow("OpenCV dilation image", img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = binary.dilate(img, kernel)
        cv2.imshow("manual dilation image", img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual

    elif mor_op == "erode":
        img_erosion = cv2.erode(img, kernel)
        cv2.imshow("OpenCV erosion image", img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = binary.erode(img, kernel)
        cv2.imshow("manual erosion image", img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual

    elif mor_op == "morph_open":
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow("OpenCV opening image", img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = binary.morph_open(img, kernel)
        cv2.imshow("manual opening image", img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    elif mor_op == "morph_close":
        img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("OpenCV closing image", img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = binary.morph_close(img, kernel)
        cv2.imshow("manual closing image", img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual

    elif mor_op == "hitmiss":
        # Calc kernel for hit_or_miss
        kernel_ = np.ones((7, 7)) * (-1)
        kernel_[1:6, 1:6] = 1

        img_hit_or_miss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel_)
        cv2.imshow("OpenCV hit_or_miss image", img_hit_or_miss)
        cv2.waitKey(wait_key_time)

        img_hit_or_miss_manual = binary.hitmiss(img, kernel_)
        cv2.imshow("manual hit_or_miss image", img_hit_or_miss_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hit_or_miss_manual

    elif mor_op == "thinning":
        img_thinning = cv2.ximgproc.thinning(img)
        cv2.imshow("OpenCV thinning image", img_thinning)
        cv2.waitKey(wait_key_time)

        img_thinning_manual = binary.thinning(img)
        cv2.imshow("manual thinning image", img_thinning_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thinning_manual

    elif mor_op == "boundary":

        img_boundary_manual = binary.boundary(img, kernel)
        cv2.imshow("manual boundary extracted image", img_boundary_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_boundary_manual

    elif mor_op == "thickening":
        img_thickening_manual = binary.thickening(img)
        cv2.imshow("manual thickening image", img_thickening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_thickening_manual

    elif mor_op == "convexhull":
        img_convex_hull_manual = binary.convex_hull(img)
        cv2.imshow("manual convex hull image", img_convex_hull_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_convex_hull_manual

    elif mor_op == "holefilling":
        fi = open(input_text_file, "r")
        points = []
        for line in fi:
            points.append(list(map(int, line.split())))
        fi.close()

        img_hole_filling_manual = binary.hole_filling(img, points)
        cv2.imshow("manual hole filling image", img_hole_filling_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_hole_filling_manual

    elif mor_op == "skeleton":
        kernel = np.ones((3, 3), np.uint8)
        img_skeleton_manual = binary.skeleton(img, kernel)
        cv2.imshow("manual skeleton image", img_skeleton_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_skeleton_manual

    elif mor_op == "extcomponent":
        components = binary.extract_component(img)
        fo = open(output_text_file, "w")

        for i in range(len(components)):
            cv2.imshow("Component no. " + str(i), components[i])
            cv2.waitKey(wait_key_time)

            index = out_file.rfind(".")
            out_file_i = out_file[:index] + "_" + str(i) + out_file[index:]
            cv2.imwrite(out_file_i, components[i])
            sum = np.sum(components[i] == 255)
            fo.write("Component no. " + str(i) + " contains: " + str(sum) + " pixels\n")

        fo.close()

    ### Gray-scale

    elif mor_op == "dilate_gray":
        img_dilation = cv2.dilate(img_gray, kernel)
        cv2.imshow("OpenCV dilation gray-scale image", img_dilation)
        cv2.waitKey(wait_key_time)

        img_dilation_manual = gray.dilate(img_gray, kernel_gray)
        cv2.imshow("manual dilation gray-scale image", img_dilation_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_dilation_manual

    elif mor_op == "erode_gray":
        img_erosion = cv2.erode(img_gray, kernel)
        cv2.imshow("OpenCV erosion gray-scale image", img_erosion)
        cv2.waitKey(wait_key_time)

        img_erosion_manual = gray.erode(img_gray, kernel_gray)
        cv2.imshow("manual erosion gray-scale image", img_erosion_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_erosion_manual

    elif mor_op == "morph_open_gray":
        img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        cv2.imshow("OpenCV opening gray-scale image", img_opening)
        cv2.waitKey(wait_key_time)

        img_opening_manual = gray.morph_open(img_gray, kernel_gray)
        cv2.imshow("manual opening gray-scale image", img_opening_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_opening_manual

    elif mor_op == "morph_close_gray":
        img_closing = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("OpenCV closing gray-scale image", img_closing)
        cv2.waitKey(wait_key_time)

        img_closing_manual = gray.morph_close(img_gray, kernel_gray)
        cv2.imshow("manual closing gray-scale image", img_closing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_closing_manual

    elif mor_op == "gradient":
        img_gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("OpenCV morphological gradient image", img_gradient)
        cv2.waitKey(wait_key_time)

        img_gradient_manual = gray.gradient(img_gray, kernel_gray)
        cv2.imshow("manual morphological gradient image", img_gradient_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_gradient_manual

    elif mor_op == "tophat":
        img_tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow("OpenCV top-hat image", img_tophat)
        cv2.waitKey(wait_key_time)

        img_tophat_manual = gray.top_hat(img_gray, kernel_gray)
        cv2.imshow("manual top-hat image", img_tophat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_tophat_manual

    elif mor_op == "blackhat":
        img_blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow("OpenCV black-hat image", img_blackhat)
        cv2.waitKey(wait_key_time)

        img_blackhat_manual = gray.black_hat(img_gray, kernel_gray)
        cv2.imshow("manual black-hat image", img_blackhat_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_blackhat_manual

    elif mor_op == "smoothing":
        img_smoothing_manual = gray.smoothing(img_gray, kernel_gray)
        cv2.imshow("manual smoothing image", img_smoothing_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_smoothing_manual

    elif mor_op == "segment":
        kernel1 = np.zeros((15, 15), np.uint8)
        kernel2 = np.zeros((100, 100), np.uint8)
        img_segment_manual = gray.segment(img_gray, kernel1, kernel2)
        cv2.imshow("manual segment image", img_segment_manual)
        cv2.waitKey(wait_key_time)

        img_out = img_segment_manual

    if img_out is not None:
        cv2.imwrite(out_file, img_out)


def main(argv):
    input_file = ""
    output_file = ""
    mor_op = ""
    wait_key_time = 0

    input_text_file = ""
    output_text_file = ""

    description = "main.py -i <input_file> -o <output_file> -p <mor_operator> -t <wait_key_time> -f <input_text_file> -j <output_text_file>"

    try:
        opts, args = getopt.getopt(
            argv,
            "hi:o:p:t:f:j:",
            [
                "in_file=",
                "out_file=",
                "mor_operator=",
                "wait_key_time=",
                "in_text_file=",
                "out_text_file=",
            ],
        )
    except getopt.GetoptError:
        print(description)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(description)
            sys.exit()
        elif opt in ("-i", "--in_file"):
            input_file = arg
        elif opt in ("-o", "--out_file"):
            output_file = arg
        elif opt in ("-p", "--mor_operator"):
            mor_op = arg
        elif opt in ("-t", "--wait_key_time"):
            wait_key_time = int(arg)
        elif opt in ("-f", "--in_text_file"):
            input_text_file = arg
        elif opt in ("-j", "--out_text_file"):
            output_text_file = arg

    print("Input file is ", input_file)
    print("Output file is ", output_file)
    print("Input text file is ", input_text_file)
    print("Output text file is ", output_text_file)
    print("Morphological operator is ", mor_op)
    print("Wait key time is ", wait_key_time)

    operator(
        input_file,
        output_file,
        mor_op,
        wait_key_time,
        input_text_file,
        output_text_file,
    )
    cv2.waitKey(wait_key_time)


if __name__ == "__main__":
    main(sys.argv[1:])
