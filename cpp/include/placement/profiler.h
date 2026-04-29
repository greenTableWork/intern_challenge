#pragma once

#if defined(PLACEMENT_ENABLE_TRACY)
#include <tracy/Tracy.hpp>
#else
#define ZoneScopedN(name) ((void)0)
#define FrameMarkNamed(name) ((void)0)
#endif
