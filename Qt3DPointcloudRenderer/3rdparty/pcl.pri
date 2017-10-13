################################################################################
## PCL
unix:!macx:include(pcl_lin_1.7.pri)
macx:include(pcl_osx_1.8.pri)
win32:include(pcl_win_1.6.pri)
