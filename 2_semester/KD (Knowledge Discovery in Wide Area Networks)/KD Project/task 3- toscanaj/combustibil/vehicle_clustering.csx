<?xml version="1.0" encoding="UTF-8"?>
<conceptualSchema version="TJ1.0">
    <databaseConnection>
        <embed url="/C:/Users/Tudor/UBB%20-%20Facultate/UBB_Master_ICA_extended/KD/ToscanaJ-1.7/examples/sql/combustibil/vehicle_cluster.sql" />
        <table name="MYTABLE" />
        <key name="ID" />
    </databaseConnection>
    <diagram title="cluster vs supplier loyalty">
        <node id="0">
            <position x="3.995119631290436" y="276.39247131347656" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="14">((CLUSTER&lt;=1)) AND ((SUPPLIERLOYALTY&gt;=3))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
            </ndimVector>
        </node>
        <node id="1">
            <position x="-95.8504947423935" y="91.13795852661133" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="3">((CLUSTER&gt;3)) AND ((SUPPLIERLOYALTY&gt;=2) AND (SUPPLIERLOYALTY&lt;3))</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="1">Supplier Share 60% - 90% </attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="2">
            <position x="98.02495634555817" y="181.22815704345703" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="8">((CLUSTER&lt;=1)) AND ((SUPPLIERLOYALTY&gt;=1) AND (SUPPLIERLOYALTY&lt;2))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
            </ndimVector>
        </node>
        <node id="3">
            <position x="99.47508335113525" y="92.45624923706055" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="9">((CLUSTER&lt;=2) AND (CLUSTER&gt;1)) AND ((SUPPLIERLOYALTY&lt;1))</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="4">Cluster 2</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="4">
            <position x="-142.40819400548935" y="138.4848976135254" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="6">((CLUSTER&gt;3)) AND ((SUPPLIERLOYALTY&gt;=3))</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="2">Supplier Share &gt; 90%</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="5">
            <position x="-91.62258666753769" y="184.17527770996094" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="1">((CLUSTER&lt;=3) AND (CLUSTER&gt;2)) AND ((SUPPLIERLOYALTY&gt;=3))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="6">
            <position x="0.0" y="0.0" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="15">((CLUSTER&gt;3)) AND ((SUPPLIERLOYALTY&lt;1))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="7">
            <position x="-45.06488740444183" y="136.82833862304688" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="10">((CLUSTER&lt;=3) AND (CLUSTER&gt;2)) AND ((SUPPLIERLOYALTY&gt;=2) AND (SUPPLIERLOYALTY&lt;3))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="8">
            <position x="50.78560733795166" y="45.69038009643555" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="11">((CLUSTER&lt;=3) AND (CLUSTER&gt;2)) AND ((SUPPLIERLOYALTY&lt;1))</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="5">Cluster 3</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="9">
            <position x="50.55281889438629" y="229.0455322265625" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="4">((CLUSTER&lt;=1)) AND ((SUPPLIERLOYALTY&gt;=2) AND (SUPPLIERLOYALTY&lt;3))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
            </ndimVector>
        </node>
        <node id="10">
            <position x="-48.37835729122162" y="43.32058334350586" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="7">((CLUSTER&gt;3)) AND ((SUPPLIERLOYALTY&gt;=1) AND (SUPPLIERLOYALTY&lt;2))</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="3">Supplier Share &lt; 60%</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="11">
            <position x="3.6245886087417603" y="183.59420776367188" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="12">((CLUSTER&lt;=2) AND (CLUSTER&gt;1)) AND ((SUPPLIERLOYALTY&gt;=2) AND (SUPPLIERLOYALTY&lt;3))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="12">
            <position x="51.096726059913635" y="135.7768325805664" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="5">((CLUSTER&lt;=2) AND (CLUSTER&gt;1)) AND ((SUPPLIERLOYALTY&gt;=1) AND (SUPPLIERLOYALTY&lt;2))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="13">
            <position x="2.4072500467300415" y="89.0109634399414" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="0">((CLUSTER&lt;=3) AND (CLUSTER&gt;2)) AND ((SUPPLIERLOYALTY&gt;=1) AND (SUPPLIERLOYALTY&lt;2))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="14">
            <position x="146.40331363677979" y="137.90757369995117" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="2">((CLUSTER&lt;=1)) AND ((SUPPLIERLOYALTY&lt;1))</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="0">Cluster 1</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
            </ndimVector>
        </node>
        <node id="15">
            <position x="-42.933110654354095" y="230.94114685058594" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="13">((CLUSTER&lt;=2) AND (CLUSTER&gt;1)) AND ((SUPPLIERLOYALTY&gt;=3))</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <edge from="9" to="0" />
        <edge from="15" to="0" />
        <edge from="10" to="1" />
        <edge from="12" to="2" />
        <edge from="14" to="2" />
        <edge from="8" to="3" />
        <edge from="1" to="4" />
        <edge from="4" to="5" />
        <edge from="7" to="5" />
        <edge from="1" to="7" />
        <edge from="13" to="7" />
        <edge from="6" to="8" />
        <edge from="2" to="9" />
        <edge from="11" to="9" />
        <edge from="6" to="10" />
        <edge from="12" to="11" />
        <edge from="7" to="11" />
        <edge from="3" to="12" />
        <edge from="13" to="12" />
        <edge from="10" to="13" />
        <edge from="8" to="13" />
        <edge from="3" to="14" />
        <edge from="5" to="15" />
        <edge from="11" to="15" />
        <projectionBase>
            <vector x="-47.472137451171875" y="47.81737518310547" />
            <vector x="0.0" y="50.0" />
            <vector x="-46.557699263095856" y="47.34693908691406" />
            <vector x="0.0" y="50.0" />
            <vector x="-44.976410269737244" y="46.0662727355957" />
            <vector x="0.0" y="50.0" />
        </projectionBase>
    </diagram>
    <diagram title="Supplier">
        <node id="0">
            <position x="-19.04997992515564" y="47.68302917480469" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="4">MYTABLE.SUPPLIER = 2</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="3">OMV</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="1">
            <position x="0.0" y="0.0" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="2">NOT (MYTABLE.SUPPLIER = 1) AND NOT (MYTABLE.SUPPLIER = 3) AND NOT (MYTABLE.SUPPLIER = 2) AND NOT (MYTABLE.SUPPLIER = 4)</object>
                </objectContingent>
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="2">
            <position x="22.305092334747314" y="44.85348129272461" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="1">MYTABLE.SUPPLIER = 1</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="1">MOL</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="3">
            <position x="70.32742595672607" y="47.139129638671875" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="0">MYTABLE.SUPPLIER = 3</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="0">Petrom</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>0.0</coordinate>
            </ndimVector>
        </node>
        <node id="4">
            <position x="5.311196953058243" y="182.42815399169922" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent />
                <attributeContingent />
            </concept>
            <ndimVector>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
                <coordinate>1.0</coordinate>
            </ndimVector>
        </node>
        <node id="5">
            <position x="-68.2713414132595" y="42.75251388549805" />
            <attributeLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </attributeLabelStyle>
            <objectLabelStyle>
                <offset x="0.0" y="0.0" />
                <backgroundColor>#ffffffff</backgroundColor>
                <textColor>#ff000000</textColor>
                <textAlignment>left</textAlignment>
            </objectLabelStyle>
            <concept>
                <objectContingent>
                    <object contextPosition="3">MYTABLE.SUPPLIER = 4</object>
                </objectContingent>
                <attributeContingent>
                    <attribute contextPosition="2">Rompetrol</attribute>
                </attributeContingent>
            </concept>
            <ndimVector>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>0.0</coordinate>
                <coordinate>1.0</coordinate>
            </ndimVector>
        </node>
        <edge from="1" to="0" />
        <edge from="1" to="2" />
        <edge from="1" to="3" />
        <edge from="0" to="4" />
        <edge from="2" to="4" />
        <edge from="3" to="4" />
        <edge from="5" to="4" />
        <edge from="1" to="5" />
        <projectionBase>
            <vector x="0.0" y="50.0" />
            <vector x="0.0" y="50.0" />
            <vector x="0.0" y="50.0" />
            <vector x="0.0" y="50.0" />
        </projectionBase>
    </diagram>
</conceptualSchema>

