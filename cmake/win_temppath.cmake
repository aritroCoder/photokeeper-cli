# OpenCV 4.10 calls GetTempPath() with a char buffer. Under UNICODE (Flutter's MSVC build) that macro
# is GetTempPathW and the compile fails. OpenCV 4.11 calls GetTempPathA; do the same until we upgrade.
file(READ modules/core/src/utils/filesystem.cpp src)
string(REPLACE "GetTempPath(" "GetTempPathA(" src "${src}")
file(WRITE modules/core/src/utils/filesystem.cpp "${src}")
