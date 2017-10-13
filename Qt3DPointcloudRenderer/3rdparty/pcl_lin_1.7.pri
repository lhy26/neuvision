################################################################################
## PCL, Boost, etc etc
LIBS += \
    # PCL
        -lpcl_common \
        -lpcl_octree \
        -lpcl_io \
        -lpcl_kdtree \
        -lpcl_search \
        -lpcl_sample_consensus \
        -lpcl_filters \
        -lpcl_segmentation \
        -lpcl_visualization \
        -lpcl_features \
        -lpcl_surface \
        -lpcl_registration \
        -lpcl_keypoints \
        -lpcl_tracking \
#        -lpcl_apps \
    # Boost
        -lboost_system \
        -lboost_filesystem \
        -lboost_thread \
#        -lboost_date_time \
#        -lboost_iostreams \

INCLUDEPATH += \
    /usr/include/pcl-1.7 \
    /usr/include/eigen3
