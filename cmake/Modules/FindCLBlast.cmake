

SET(CL_BLAST_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/local/include
  /opt/CLBlast/include
  $ENV{CLBlast_HOME}
  $ENV{CLBlast_HOME}/include
)

SET(CL_BLAST_LIB_SEARCH_PATHS
        /lib/
        /lib64/
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/CLBlast/lib
        $ENV{CLBlast}cd
        $ENV{CLBlast}/lib
        $ENV{CLBlast_HOME}
        $ENV{CLBlast_HOME}/lib
 )

FIND_PATH(CLBlast_INCLUDE_DIR NAMES clblast_c.h PATHS ${CL_BLAST_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(CLBlast_LIB NAMES clblast PATHS ${CL_BLAST_LIB_SEARCH_PATHS})

SET(CLBLAST_FOUND ON)

#    Check include files
IF(NOT CLBlast_INCLUDE_DIR)
    SET(CLBLAST_FOUND OFF)
    MESSAGE(STATUS "Could not find CLBlast include. Turning CLBLAST_FOUND off")
ENDIF()

#    Check libraries
IF(NOT CLBlast_LIB)
    SET(CLBLAST_FOUND OFF)
    MESSAGE(STATUS "Could not find CLBlast lib. Turning CLBLAST_FOUND off")
ENDIF()

IF (CLBLAST_FOUND)
  IF (NOT CLBLAST_FIND_QUIETLY)
    MESSAGE(STATUS "Found CLBlast libraries: ${CLBlast_LIB}")
    MESSAGE(STATUS "Found CLBlast include: ${CLBlast_INCLUDE_DIR}")
  ENDIF (NOT CLBLAST_FIND_QUIETLY)
ELSE (CLBLAST_FOUND)
  IF (CLBlast_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find CLBlast")
  ENDIF (CLBlast_FIND_REQUIRED)
ENDIF (CLBLAST_FOUND)

MARK_AS_ADVANCED(
    CLBlast_INCLUDE_DIR
    CLBlast_LIB
    CLBlast
)

