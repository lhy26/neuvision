#include "quickVtkViewer.hpp"

#include "quickVtkFboRenderer.hpp"
#include "quickVtkFboOffscreenWindow.hpp"

#include <QOpenGLFunctions>
#include <QQuickFramebufferObject>
#include <QOpenGLFramebufferObject>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRendererCollection.h>
#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

Viewer::Viewer()
    : QQuickFramebufferObject()
    , m_renderer(nullptr)
    , m_win(FboOffscreenWindow::New())
{
	qDebug() << Q_FUNC_INFO;

	setMirrorVertically(true);

	m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_renderer->SetBackground2(0.7, 0.7, 0.7);
    m_renderer->SetBackground(0.6, 0.6, 0.6);
	m_renderer->GradientBackgroundOn();

	m_win->AddRenderer(m_renderer);
}

Viewer::~Viewer()
{
	qDebug() << Q_FUNC_INFO;

	m_win->Delete();
}

auto Viewer::init() -> void
{
	qDebug() << Q_FUNC_INFO;

//	auto rw = GetRenderWindow();
//	m_renderer = vtkSmartPointer<vtkRenderer>::New();
//	rw->AddRenderer(m_renderer);
	m_initialized = true;

	update();
}

auto Viewer::update() -> void
{
	qDebug() << Q_FUNC_INFO;

	if (!m_initialized) {
		return;
	}

	if (m_fboRenderer) {
		QQuickFramebufferObject::update();
	}
}

auto Viewer::setHoverEnabled(bool hoverEnabled) -> void
{
	m_hoverEnabled = hoverEnabled;

	setAcceptHoverEvents(hoverEnabled);

	emit hoverEnabledChanged();
}

auto Viewer::setMouseEnabled(bool mouseEnabled) -> void
{
	m_mouseEnabled = mouseEnabled;

	if (mouseEnabled) {
		setAcceptedMouseButtons(Qt::AllButtons);
	} else {
		setAcceptedMouseButtons(Qt::NoButton);
	}

	emit mouseEnabledChanged();
}

auto Viewer::mousePressEvent(QMouseEvent* event) -> void
{
	if (m_fboRenderer) {
		m_fboRenderer->onMouseEvent(event);
	}
}

auto Viewer::mouseReleaseEvent(QMouseEvent* event) -> void
{
	if (m_fboRenderer) {
		m_fboRenderer->onMouseEvent(event);
	}
}

auto Viewer::mouseMoveEvent(QMouseEvent* event) -> void
{
	if (m_fboRenderer) {
		m_fboRenderer->onMouseEvent(event);
	}
}

auto Viewer::hoverMoveEvent(QHoverEvent* event) -> void
{

}

auto Viewer::createRenderer() const -> QQuickFramebufferObject::Renderer*
{
	qDebug() << Q_FUNC_INFO;

	m_fboRenderer = new FboRenderer(static_cast<FboOffscreenWindow*>(m_win));

	return m_fboRenderer;
}

auto Viewer::GetRenderWindow() const -> vtkGenericOpenGLRenderWindow*
{
	qDebug() << Q_FUNC_INFO;

	return m_win;
}

auto Viewer::getHoverEnabled() -> bool
{
	return m_hoverEnabled;
}

auto Viewer::getMouseEnabled() -> bool
{
	return m_mouseEnabled;
}
