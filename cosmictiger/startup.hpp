/*
 CosmicTiger - A cosmological N-Body code
 Copyright (C) 2026  Dominic C. Marcello

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <string_view>

namespace cosmictiger {

inline constexpr std::string_view kStartupScreen = R"(                                              \  |  /                     *----*
                      .             ___________\ | /                      |    |
                         __________/____________\|/                   *---*----*
            ____________/________________________*----             .-'
       ____/                                   / | \            *-'
             .                                /  |  \        .-'
     *                   .                                  *

                 /\               /\
                /  \_____________/  \_____
               /  //\    |||    /\\   \    '--.__
              |  //  \   |||   /  \\  |  //  // '--._           ___
              | || ==(^)     (^)== || | //  //  //   '-.      .'   '.
              | ||       \_/      || |/  //  //       \    / .--.  \
              |  \\  \ V     V /  //  |   \\  \\       |   | |    | |
               \      \_______/      /    \\  \\      |   | |    | |
                \                   /      \\  \\     /   | |    | |
                 '._            _.'  ____         _.'    | |   / /
                    '-._____.-'_____/    \_______/  \___/  '--' /
               _____/        \___         \    \   |   \_____.-'
         _____/    /    \        \_________\____\__/       \
        /  __/____/      \____    \_______/       /________/
       (___/___/___)      (___/___/___)   (___/___/___/___)

  ____   ___   ____   __  __  ___   ____        _____  ___   ____  _____  ____
 / ___| / _ \ / ___| |  \/  ||_ _| / ___|      |_   _||_ _| / ___|| ____||  _ \
| |    | | | |\___ \ | |\/| | | | | |            | |   | | | |  _ |  _|  | |_) |
| |___ | |_| | ___) || |  | | | | | |___         | |   | | | |_| || |___ |  _ <
 \____| \___/ |____/ |_|  |_||___| \____|        |_|  |___| \____||_____||_| \_\

                           A cosmological N-Body code
)";

}
