#include "quickVtkFboOffscreenWindow.hpp"
#include "quickVtkFboRenderer.hpp"

FboOffscreenWindow::FboOffscreenWindow()
    : vtkExternalOpenGLRenderWindow()
    , QOpenGLFunctions()
    , QtParentRenderer(nullptr)
{
	qDebug() << Q_FUNC_INFO;

	OffScreenRenderingOn();
}

FboOffscreenWindow::~FboOffscreenWindow()
{
	qDebug() << Q_FUNC_INFO;

	OffScreenRendering = 0;
}

auto FboOffscreenWindow::New() -> FboOffscreenWindow*
{
	qDebug() << Q_FUNC_INFO;

	return new FboOffscreenWindow();
}

auto FboOffscreenWindow::OpenGLInitState() -> void
{
    qDebug() << Q_FUNC_INFO;

	Superclass::OpenGLInitState();
/*
	MakeCurrent();
	initializeOpenGLFunctions();

	Superclass::OpenGLInitState();
	glUseProgram(0);

	glEnable(GL_BLEND);
	glHint(GL_CLIP_VOLUME_CLIPPING_HINT_EXT, GL_FASTEST);
    glDepthMask(GL_TRUE);*/
}

auto FboOffscreenWindow::Render() -> void
{
	qDebug() << Q_FUNC_INFO;

    if (QtParentRenderer) {
        QtParentRenderer->update();
    }
}

auto FboOffscreenWindow::InternalRender() -> void
{
    qDebug() << Q_FUNC_INFO;

    QMutexLocker lock(&mutex);

    Superclass::Render();
}

auto FboOffscreenWindow::SetFramebufferObject(QOpenGLFramebufferObject *fbo) -> void
{
	qDebug() << Q_FUNC_INFO;

	BackLeftBuffer = FrontLeftBuffer = BackBuffer = FrontBuffer = static_cast<unsigned int>(GL_COLOR_ATTACHMENT0);

	auto size                 = fbo->size();

	Size[0]                   = size.width();
	Size[1]                   = size.height();
	NumberOfFrameBuffers      = 1;
	FrameBufferObject         = static_cast<unsigned int>(fbo->handle());
	DepthRenderBufferObject   = 0;
	TextureObjects[0]         = static_cast<unsigned int>(fbo->texture());
	OffScreenRendering        = 1;
	OffScreenUseFrameBuffer   = 1;
	Modified();
}
