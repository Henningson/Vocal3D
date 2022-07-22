from PyQt5.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QLabel, QCheckBox, QLineEdit
from QLines import QHLine


class SubMenuWidget(QWidget):
    def __init__(self, title, listOfTriplets, parent=None):
        super(SubMenuWidget, self).__init__()
        base_layout = QVBoxLayout(self)

        title = QLabel(title)
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)

        base_layout.addWidget(title)
        
        base_layout.addWidget(QHLine())
        
        self.DICT = {}

        self.subform = QWidget(self)
        self.subform_layout = QFormLayout()
        self.subform.setLayout(self.subform_layout)

        for key, button_type, default_value in listOfTriplets:
            if button_type == "checkbox":
                check = QCheckBox(self.subform)
                check.setChecked(default_value)
                self.subform_layout.addRow(QLabel(key, self.subform), check)
                self.DICT[key] = check
            elif button_type == "field":
                lineedit = QLineEdit(str(default_value), self.subform)
                self.subform_layout.addRow(QLabel(key, self.subform), lineedit)
                self.DICT[key] = lineedit

        base_layout.addWidget(self.subform)

    def get_dict(self):
        return self.DICT