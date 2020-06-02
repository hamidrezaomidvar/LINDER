<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" maxScale="0" version="3.4.5-Madeira" hasScaleBasedVisibilityFlag="0" minScale="1e+08">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <customproperties>
    <property key="WMSBackgroundLayer" value="false"/>
    <property key="WMSPublishDataSourceUrl" value="false"/>
    <property key="embeddedWidgets/count" value="0"/>
    <property key="identify/format" value="Value"/>
  </customproperties>
  <pipe>
    <rasterrenderer opacity="0.514" band="1" classificationMin="1.1" type="singlebandpseudocolor" classificationMax="9.3" alphaBand="-1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader classificationMode="1" colorRampType="INTERPOLATED" clip="0">
          <colorramp name="[source]" type="gradient">
            <prop k="color1" v="215,25,28,255"/>
            <prop k="color2" v="43,131,186,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.25;253,174,97,255:0.5;255,255,191,255:0.75;171,221,164,255"/>
          </colorramp>
          <item color="#1f78b4" label="Inland Water" value="1.1" alpha="255"/>
          <item color="#1f78b4" label="Coastal water" value="1.2" alpha="255"/>
          <item color="#fdd602" label="Coastal dunes" value="2.1" alpha="255"/>
          <item color="#515151" label="Natural rockscapes" value="2.2" alpha="255"/>
          <item color="#d95eae" label="Open or heath and moor land" value="2.3" alpha="255"/>
          <item color="#4cf7fc" label="Wetlands" value="2.4" alpha="255"/>
          <item color="#f1ca92" label="Agriculture - mainly crops" value="3.1" alpha="255"/>
          <item color="#7afc74" label="Agriculture - mixed use" value="3.2" alpha="255"/>
          <item color="#a714fc" label="Vineyards and hopyards" value="3.3" alpha="255"/>
          <item color="#cff9d8" label="Glasshouses" value="3.4" alpha="255"/>
          <item color="#1ccd45" label="Orchards" value="3.5" alpha="255"/>
          <item color="#85997e" label="Farms" value="3.6" alpha="255"/>
          <item color="#33a02c" label="Deciduous woodland" value="4.1" alpha="255"/>
          <item color="#174914" label="Coniferous and undifferentiated woodland" value="4.2" alpha="255"/>
          <item color="#dadada" label="Principle Transport" value="5.1" alpha="255"/>
          <item color="#fb0699" label="Mining and spoil areas" value="6.1" alpha="255"/>
          <item color="#cbf161" label="Recreational land" value="7.1" alpha="255"/>
          <item color="#ff7f00" label="Large complex buildings various use (travel/recreation/ retail)&#xa;" value="7.2" alpha="255"/>
          <item color="#e6d879" label="Low density residential with amenities (suburbs and small villages / hamlets)" value="8.1" alpha="255"/>
          <item color="#fb9a99" label="Medium density residential with high streets and amenities&#xa;" value="8.2" alpha="255"/>
          <item color="#c31911" label="High density residential with retail and commercial sites&#xa;" value="8.3" alpha="255"/>
          <item color="#5a0403" label="Urban centres - mainly commercial/retail with residential pockets&#xa;" value="8.4" alpha="255"/>
          <item color="#9d4c00" label="Industrial Areas" value="9.1" alpha="255"/>
          <item color="#ffa2ff" label="Business parks" value="9.2" alpha="255"/>
          <item color="#cba2ff" label="Retail parks" value="9.3" alpha="255"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation colorizeRed="255" grayscaleMode="0" saturation="0" colorizeOn="0" colorizeStrength="100" colorizeGreen="128" colorizeBlue="128"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
