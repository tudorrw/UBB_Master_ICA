module com.example.mas {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;
    requires com.dlsc.formsfx;
    requires jade;

    opens com.example.mas to javafx.fxml;
    exports com.example.mas;

    opens com.example.mas.agents to javafx.fxml;
    exports com.example.mas.agents;

    exports com.example.mas.entities;
    opens com.example.mas.entities to javafx.fxml;
    exports com.example.mas.agents.ambulance;
    opens com.example.mas.agents.ambulance to javafx.fxml;

}