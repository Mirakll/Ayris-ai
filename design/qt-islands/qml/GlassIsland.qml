import QtQuick
import QtQuick.Effects

// Остров «Стекло / Aurora»: живой цветной фон → настоящий GPU-блюр (MultiEffect)
// → полупрозрачная стеклянная карточка с рамкой. Наведение поднимает карточку
// через Behavior — то, что в QSS не сделать. Все цвета/радиусы/тайминги — токены.
Item {
    id: root
    implicitWidth: 460
    implicitHeight: 340

    // 1) сцена, которая «просвечивает» сквозь стекло (её и блюрим)
    Item {
        id: backdrop
        anchors.fill: parent
        layer.enabled: true            // рендер в текстуру для MultiEffect
        layer.smooth: true

        Rectangle { anchors.fill: parent; color: theme.background }

        Rectangle {                    // коралловое/акцентное пятно
            width: 240; height: 240; radius: 120
            color: theme.accent; opacity: 0.85
            SequentialAnimation on x {
                loops: Animation.Infinite
                NumberAnimation { from: 20; to: root.width - 200; duration: 4200; easing.type: Easing.InOutSine }
                NumberAnimation { from: root.width - 200; to: 20; duration: 4200; easing.type: Easing.InOutSine }
            }
            y: 30
        }
        Rectangle {                    // информационное пятно
            width: 280; height: 280; radius: 140
            color: theme.info; opacity: 0.75
            x: root.width - 220
            SequentialAnimation on y {
                loops: Animation.Infinite
                NumberAnimation { from: root.height - 240; to: 24; duration: 5200; easing.type: Easing.InOutSine }
                NumberAnimation { from: 24; to: root.height - 240; duration: 5200; easing.type: Easing.InOutSine }
            }
        }
    }

    // 2) НАСТОЯЩИЙ блюр всей сцены на GPU (Qt 6.5+)
    MultiEffect {
        anchors.fill: parent
        source: backdrop
        blurEnabled: true
        blur: 1.0
        blurMax: 48
        saturation: 0.15
    }

    // 3) стеклянная карточка поверх блюра
    Item {
        id: card
        width: parent.width * 0.64
        height: parent.height * 0.52
        anchors.centerIn: parent
        scale: hover.hovered ? 1.03 : 1.0
        Behavior on scale { NumberAnimation { duration: theme.animNormal; easing.type: Easing.OutBack } }

        Rectangle {
            anchors.fill: parent
            radius: theme.radiusXl
            color: Qt.rgba(theme.surface.r, theme.surface.g, theme.surface.b, 0.42)
            border.width: 1
            border.color: hover.hovered ? theme.accent : Qt.rgba(1, 1, 1, 0.16)
            Behavior on border.color { ColorAnimation { duration: theme.animNormal } }

            Column {
                anchors.centerIn: parent
                spacing: 8
                Text {
                    text: "Стекло / Aurora"
                    color: theme.textPrimary
                    font.pixelSize: 22; font.weight: Font.DemiBold
                    anchors.horizontalCenter: parent.horizontalCenter
                }
                Text {
                    text: "backdrop-blur без эмуляции — MultiEffect на GPU"
                    color: theme.textSecondary
                    font.pixelSize: 13
                    anchors.horizontalCenter: parent.horizontalCenter
                }
            }
        }
        HoverHandler { id: hover }
    }
}
