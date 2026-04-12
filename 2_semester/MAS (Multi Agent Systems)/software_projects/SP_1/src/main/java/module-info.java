module com.example.mas {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires jade;

    opens com.example.mas to javafx.fxml;
    exports com.example.mas;
}