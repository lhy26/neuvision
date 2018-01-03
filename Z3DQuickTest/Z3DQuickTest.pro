include(../NEUVision.pri)

QT += quick opengl

#CONFIG += console

DESTDIR = $$Z3D_BUILD_DIR

HEADERS += \
#    QVTKFramebufferObjectItem.h \
    quickVtkFboOffscreenWindow.hpp \
    quickVtkFboRenderer.hpp \
    quickVtkViewer.hpp \
    quickVtkWin32Interactor.hpp

SOURCES += main.cpp \
#    QVTKFramebufferObjectItem.cpp \
    quickVtkWin32Interactor.cpp \
    quickVtkViewer.cpp \
    quickVtkFboOffscreenWindow.cpp \
    quickVtkFboRenderer.cpp

RESOURCES += qml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

#################################################################################
### VTK
include($$PWD/../3rdparty/vtk.pri)

###############################################################################
# Core
include($$PWD/../lib/zcore/zcore.pri)

###############################################################################
# Gui
include($$PWD/../lib/zgui/zgui.pri)
