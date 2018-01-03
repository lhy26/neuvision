#include "quickVtkViewer.hpp"

#include <zapplication.h>

#if defined(Q_OS_MACOS)
#include "osxutils.h"
#endif

//#include <QApplication>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QTimer>
#include <QWindow>

#include "vtkAutoInit.h"
VTK_MODULE_INIT(vtkRenderingOpenGL2)
VTK_MODULE_INIT(vtkInteractionStyle)

#include <QVTKOpenGLWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkConeSource.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkPolyDataMapper.h>

#include <vtkPolyData.h>
#include <vtkPLYReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkLODActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkGlyph3DMapper.h>
#include <vtkRegularPolygonSource.h>
#include <vtkDataSetAttributes.h>
#include <vtkProperty.h>
#include "vtkActor.h"
#include "vtkCamera.h"
#include "vtkColorTransferFunction.h"
#include "vtkDataObject.h"
#include "vtkDataSetAttributes.h"
#include "vtkNew.h"
#include "vtkPiecewiseFunction.h"
#include "vtkPointGaussianMapper.h"
#include "vtkPointCloudFilter.h"
#include "vtkPointSource.h"
#include "vtkProperty.h"
#include "vtkRandomAttributeGenerator.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkSphereSource.h"
#include "vtkTimerLog.h"

#include <vtkVertexGlyphFilter.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkPolyDataNormals.h>

int main(int argc, char *argv[])
{
/*
    if (argc != 2) {
		std::cout << "Usage: " << argv[0] << "  Filename(.ply)" << std::endl;
		return EXIT_FAILURE;
	}

	std::string inputFilename = argv[1];
*/





	// before initializing QApplication, set the default surface format.
    QSurfaceFormat format = QVTKOpenGLWidget::defaultFormat();
    format.setSamples(0);
    QSurfaceFormat::setDefaultFormat(format);
//  QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    Z3D::ZApplication app(argc, argv);

	qmlRegisterType<Viewer>("QuickVTK", 1, 0, "VtkViewer");

	QQmlApplicationEngine engine;
	engine.load(QUrl(QLatin1String("qrc:/main.qml")));

	if (engine.rootObjects().isEmpty()) {
		return -1;
	}

    QObject *rootObject = engine.rootObjects().first();
    QWindow *window = static_cast<QWindow*>(rootObject);

#if defined(Q_OS_MACOS)
//    osxHideTitleBar(window->winId());
    osxTransparentTitleBar(window->winId());
#endif

	QList<Viewer*> vtkItems = window->findChildren<Viewer*>();

	QTimer::singleShot(100, [=](){
		// For demonstration: Add a cone to the scene of each Viewer

		vtkNew<vtkConeSource> polyDataSource;
		polyDataSource->SetResolution(100);
		vtkNew<vtkPolyDataMapper> mapper;
		vtkNew<vtkActor> actor;
		mapper->SetInputConnection(polyDataSource->GetOutputPort());
		actor->SetMapper(mapper);
		actor->GetProperty()->SetOpacity(0.5); // demonstrate support for translucent VTK objects


/*
		vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
		reader->SetFileName( inputFilename.c_str() );
		reader->Update();
		auto polyData = reader->GetOutput();

		/// generate normals
		vtkSmartPointer<vtkPolyDataNormals> normalGenerator = vtkSmartPointer<vtkPolyDataNormals>::New();
		normalGenerator->SetInputData(polyData);
		normalGenerator->ComputePointNormalsOn();
		normalGenerator->ComputeCellNormalsOff();
		normalGenerator->Update();

		polyData = normalGenerator->GetOutput();

		std::cout << "points: " << polyData->GetNumberOfPoints() << std::endl;
		std::cout << "polys: " << polyData->GetNumberOfPolys() << std::endl;

		vtkNew<vtkOpenGLPolyDataMapper> mapper;
		mapper->SetInputData(polyData);

		vtkNew<vtkActor> actor;
		actor->SetMapper(mapper);
*/
		for (auto *vtkItem : vtkItems) {
			auto *renderWindow = vtkItem->GetRenderWindow();
			auto *renderer = renderWindow->GetRenderers()->GetFirstRenderer();

			if (!renderer) {
				qWarning() << "renderer is null";
				return;
			}

			qDebug() << "adding actor";

			renderer->AddActor(actor);
			renderer->ResetCamera();
			renderer->ResetCameraClippingRange();

			vtkItem->update();
		}
	});

	return app.exec();
}
