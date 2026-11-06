# STARSDataFusion.jl

Spatial Timeseries for Automated high-Resolution multi-Sensor data fusion (STARS) Julia Package
<!-- 
Margaret C. Johnson (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
Principal investigator: lead of data fusion methodological development and Julia code implementations.<br>
NASA Jet Propulsion Laboratory 398L

Gregory H. Halverson (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer for data processing pipeline design and development, moving window implementation, and code organization and management.<br>
NASA Jet Propulsion Laboratory 329G

Jouni I. Susiluoto<br>
[jouni.i.susiluoto@jpl.nasa.gov](mailto:jouni.i.susiluoto@jpl.nasa.gov)<br>
Technical contributor for methodology development, co- developer of Julia code for Kalman filtering recursion.
NASA Jet Propulsion Laboratory 398L

Kerry Cawse-Nicholson (she/her)<br>
[kerry-anne.cawse-nicholson@jpl.nasa.gov](mailto:kerry-anne.cawse-nicholson@jpl.nasa.gov)<br>
Concept development and project management. Advised on technical and scientific requirements for application and mission integration.<br>
NASA Jet Propulsion Laboratory 329G

Joshua B. Fisher (he/him)<br>
[jbfisher@chapman.edu](mailto:jbfisher@chapman.edu)<br>
Concept development and project management<br>
Chapman University

Glynn C. Hulley (he/him)<br>
[glynn.hulley@jpl.nasa.gov](mailto:glynn.hulley@jpl.nasa.gov)<br>
Advised on technical and scientific requirements for application and mission integration.<br>
NASA Jet Propulsion Laboratory 329G

Nimrod Carmon (he/him)<br>
[nimrod.carmon@jpl.nasa.gov](mailto:nimrod.carmon@jpl.nasa.gov)<br>
Technical contributor for data processing, validation/verification, and hyperspectral resampling<br>
NASA Jet Propulsion Laboratory 398L
-->

STARS is a general data fusion methodology utilizing spatiotemporal statistical models to optimally combine high spatial resolution VSWIR measurements with high temporal resolution measurements from multiple instruments. The methods are highly-scalable, able to fuse <100 m spatial resolution products in near-real time (<24 hrs) on regional to global scales, to facilitate online data processing as well as large-scale reprocessing of mission datasets. The statistical spatiotemporal modeling framework provides with each fused surface reflectance product associated pixel-level uncertainties incorporating any known data source measurement uncertainties, bias characteristics, and degree of historical data missingness. 

The specific capabilities offered by STARS are: 
1. automatic, high-resolution spatial and temporal gap-filling, 
2. a tunable fusion framework allowing the user to choose a level of accuracy vs computational complexity, and 
3. quantifiable uncertainties that can be used for downstream product sensitivity/uncertainty assessments and that can be incorporated into higher-order data product quality flags. 


