#pragma once

#include <QOpenGLFunctions>
#include <QOpenGLFramebufferObject>

#include <vtkExternalOpenGLRenderWindow.h>

class FboRenderer;

class FboOffscreenWindow : public vtkExternalOpenGLRenderWindow, protected QOpenGLFunctions
{
public:
	static auto New() -> FboOffscreenWindow*;

	virtual auto OpenGLInitState() -> void override;
	auto Render() -> void override;
	auto InternalRender() -> void;
	auto SetFramebufferObject(QOpenGLFramebufferObject*) -> void;

	FboRenderer* QtParentRenderer;

protected:
	FboOffscreenWindow();
	~FboOffscreenWindow();

    QMutex mutex;
};
