//
// Z3D - A structured light 3D scanner
// Copyright (C) 2013-2016 Nicolas Ulrich <nikolaseu@gmail.com>
//
// This file is part of Z3D.
//
// Z3D is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Z3D is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Z3D.  If not, see <http://www.gnu.org/licenses/>.
//

#include "zbinarypatternprojectionplugin.h"
#include "zbinarypatternprojection.h"
#include "zbinarypatternprojectionconfigwidget.h"

namespace Z3D
{

ZBinaryPatternProjectionPlugin::ZBinaryPatternProjectionPlugin()
{

}

QWidget *ZBinaryPatternProjectionPlugin::getConfigWidget(ZPatternProjectionWeakPtr patternProjection)
{
    const auto item = m_patternProjectionWidgets.find(patternProjection);
    if (item != m_patternProjectionWidgets.end()) {
        return item->second;
    }

    QWidget *widget = nullptr;
    if (auto *binaryPatternProjection = qobject_cast<ZBinaryPatternProjection*>(patternProjection)) {
        widget = new ZBinaryPatternProjectionConfigWidget(binaryPatternProjection);
    }

    if (widget) {
        QObject::connect(patternProjection, &QObject::destroyed, [=](QObject *) {
            m_patternProjectionWidgets.erase(patternProjection);
        });
        m_patternProjectionWidgets[patternProjection] = widget;
    }

    return widget;
}

ZPatternProjectionPtr ZBinaryPatternProjectionPlugin::get(QSettings *settings)
{
    Q_UNUSED(settings)

    return ZPatternProjectionPtr(new ZBinaryPatternProjection());
}

} // namespace Z3D
