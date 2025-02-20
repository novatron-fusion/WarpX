find_package(Boost 1.66.0 COMPONENTS headers)

foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        EmbeddedBoundaryInit.cpp
        Enabled.cpp
        WarpXInitEB.cpp
        WarpXFaceExtensions.cpp
        WarpXFaceInfoBox.H
        PolygonXYIF.H
    )
endforeach()
