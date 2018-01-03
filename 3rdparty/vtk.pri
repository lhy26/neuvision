################################################################################
## VTK
################################################################################
win32:VTK_DIR=C:\Builds\VTK-8.1.0-install
macx:VTK_DIR=/usr/local/opt/vtk

VTK_VERSION=8.1

LIBS += \
    -L$$VTK_DIR/lib \
        -lvtkCommonCore-$$VTK_VERSION \
        -lvtkCommonExecutionModel-$$VTK_VERSION \
        -lvtkCommonDataModel-$$VTK_VERSION \
        -lvtkCommonMath-$$VTK_VERSION \
        -lvtkCommonSystem-$$VTK_VERSION \
        -lvtkGUISupportQt-$$VTK_VERSION \
        -lvtkIOImage-$$VTK_VERSION \
        -lvtkIOPly-$$VTK_VERSION \
        -lvtkRenderingCore-$$VTK_VERSION \
        -lvtkRenderingLOD-$$VTK_VERSION \
        -lvtkRenderingAnnotation-$$VTK_VERSION \
        -lvtkRenderingContextOpenGL2-$$VTK_VERSION \
        -lvtkRenderingExternal-$$VTK_VERSION \
        -lvtkRenderingOpenGL2-$$VTK_VERSION \
        -lvtkRenderingQt-$$VTK_VERSION \
        -lvtkFiltersCore-$$VTK_VERSION \
        -lvtkFiltersGeneral-$$VTK_VERSION \
        -lvtkFiltersSources-$$VTK_VERSION \
        -lvtkFiltersGeometry-$$VTK_VERSION \
        -lvtkInteractionStyle-$$VTK_VERSION \

INCLUDEPATH += \
    $$VTK_DIR/include/vtk-$$VTK_VERSION/ \
