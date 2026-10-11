import QtQuick
import QtQuick.Effects

// Остров «Неон-киберпанк»: сфера состояния с неоновым свечением (halo = блюр
// яркой копии за чётким орбом, MultiEffect) + дыхание/пульс через анимации.
// Свечение и bloom — то, что QSS изобразить не может; здесь это нативно.
Item {
    id: root
    implicitWidth: 460
    implicitHeight: 340

    property color glow: theme.accent

    Rectangle { anchors.fill: parent; color: theme.background }

    // сетка-синтвейв (лёгкая, чтобы читалась «сцена»)
    Row {
        anchors.fill: parent
        spacing: 34
        Repeater {
            model: 16
            Rectangle { width: 1; height: root.height; color: Qt.rgba(theme.accent.r, theme.accent.g, theme.accent.b, 0.10) }
        }
    }

    // чёткий орб — источник свечения
    Rectangle {
        id: orb
        width: 150; height: 150; radius: 75
        anchors.centerIn: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.lighter(root.glow, 1.5) }
            GradientStop { position: 1.0; color: Qt.darker(root.glow, 1.4) }
        }
        border.width: 2
        border.color: Qt.rgba(1, 1, 1, 0.5)
        layer.enabled: true

        // дыхание
        SequentialAnimation on scale {
            loops: Animation.Infinite
            NumberAnimation { from: 1.0; to: 1.04; duration: 2400; easing.type: Easing.InOutSine }
            NumberAnimation { from: 1.04; to: 1.0; duration: 2400; easing.type: Easing.InOutSine }
        }
    }

    // halo = блюр копии орба ПОД ним → неоновое свечение
    MultiEffect {
        anchors.centerIn: parent
        width: orb.width * 2.2
        height: orb.height * 2.2
        source: orb
        blurEnabled: true
        blur: 1.0
        blurMax: 64
        brightness: 0.25
        colorization: 0.6
        colorizationColor: root.glow
        z: -1
        // пульс свечения
        SequentialAnimation on brightness {
            loops: Animation.Infinite
            NumberAnimation { from: 0.15; to: 0.45; duration: 3000; easing.type: Easing.InOutSine }
            NumberAnimation { from: 0.45; to: 0.15; duration: 3000; easing.type: Easing.InOutSine }
        }
    }

    // реакция на клик = короткий bounce + вспышка свечения (как «речь»)
    TapHandler {
        onTapped: { bounce.restart(); flash.restart() }
    }
    SequentialAnimation {
        id: bounce
        NumberAnimation { target: orb; property: "scale"; to: 1.12; duration: 120; easing.type: Easing.OutQuad }
        NumberAnimation { target: orb; property: "scale"; to: 1.0; duration: 220; easing.type: Easing.OutBack }
    }
    SequentialAnimation {
        id: flash
        ColorAnimation { target: root; property: "glow"; to: theme.info; duration: 160 }
        ColorAnimation { target: root; property: "glow"; to: theme.accent; duration: 600 }
    }

    Text {
        anchors { horizontalCenter: parent.horizontalCenter; bottom: parent.bottom; bottomMargin: 22 }
        text: "Неон · клик = импульс"
        color: theme.textSecondary
        font.pixelSize: 13
    }
}
