#include "quickVtkFboRenderer.hpp"
#include "quickVtkFboOffscreenWindow.hpp"
#include "quickVtkViewer.hpp"

#ifdef _MSC_VER
#include "quickVtkWin32Interactor.hpp"
#endif

#include <vtkRendererCollection.h>
#include <vtkCamera.h>
#include <vtkCommand.h>

FboRenderer::FboRenderer(FboOffscreenWindow *offscreenWindow)
    : QQuickFramebufferObject::Renderer()
    , m_fboOffscreenWindow(offscreenWindow)
    , m_fbo(nullptr)
{
	qDebug() << Q_FUNC_INFO;

	m_fboOffscreenWindow->Register(nullptr);
	m_fboOffscreenWindow->QtParentRenderer = this;

#ifndef _MSC_VER
	m_interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	m_interactorStyle = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();

	m_interactor->SetRenderWindow(offscreenWindow);
	m_interactor->Initialize();
	m_interactor->SetInteractorStyle(m_interactorStyle);
#else
	m_interactor = vtkSmartPointer<Win32Interactor>::New();
	m_interactorStyle = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();

	m_interactor->SetRenderWindow(offscreenWindow);
	m_interactor->Initialize();
	m_interactor->SetInteractorStyle(m_interactorStyle);
#endif
}

FboRenderer::~FboRenderer()
{
	qDebug() << Q_FUNC_INFO;

	m_fboOffscreenWindow->QtParentRenderer = nullptr;
	m_fboOffscreenWindow->Delete();
}

auto FboRenderer::synchronize(QQuickFramebufferObject* fbo) -> void
{
	qDebug() << Q_FUNC_INFO;

	if (!m_fbo) {
		auto viewer = static_cast<Viewer*>(fbo);
		viewer->init();
	}
}

auto FboRenderer::render() -> void
{
	qDebug() << Q_FUNC_INFO;

//	m_fboOffscreenWindow->PushState();
//	m_fboOffscreenWindow->OpenGLInitState();
	m_fboOffscreenWindow->InternalRender();
//	m_fboOffscreenWindow->PopState();
}

auto FboRenderer::createFramebufferObject(const QSize &size) -> QOpenGLFramebufferObject*
{
	qDebug() << Q_FUNC_INFO;

	QOpenGLFramebufferObjectFormat format;
	format.setAttachment(QOpenGLFramebufferObject::Depth);
	m_fbo = new QOpenGLFramebufferObject(size, format);
	m_fboOffscreenWindow->SetFramebufferObject(m_fbo);

	return m_fbo;
}

auto FboRenderer::onMouseEvent(QMouseEvent* event) -> void
{
	qDebug() << Q_FUNC_INFO;

    if (m_interactor == nullptr || event == nullptr) {
		return;
	}

    if (!m_interactor->GetEnabled()) {
		return;
	}

	m_interactor->SetEventInformationFlipY(event->x(), event->y(), (event->modifiers() & Qt::ControlModifier), (event->modifiers() & Qt::ShiftModifier));

	auto command = vtkCommand::NoEvent;

	if (event->type() == QEvent::MouseMove) {
		command = vtkCommand::MouseMoveEvent;
	}

	else if (event->type() == QEvent::MouseButtonPress || event->type() == QEvent::MouseButtonDblClick) {
		switch (event->button()) {
		case Qt::LeftButton:    command = vtkCommand::LeftButtonPressEvent; break;
		case Qt::RightButton:   command = vtkCommand::RightButtonPressEvent; break;
		case Qt::MidButton:     command = vtkCommand::MiddleButtonPressEvent; break;
		default: break;
		}
	}

	else if (event->type() == QEvent::MouseButtonRelease) {
		switch (event->button()) {
		case Qt::LeftButton:    command = vtkCommand::LeftButtonReleaseEvent; break;
		case Qt::RightButton:   command = vtkCommand::RightButtonReleaseEvent; break;
		case Qt::MidButton:     command = vtkCommand::MiddleButtonReleaseEvent; break;
		default: break;
		}
	}

    m_interactor->InvokeEvent(command, event);
}
