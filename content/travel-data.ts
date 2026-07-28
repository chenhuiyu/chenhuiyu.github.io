export type TravelLocation = {
  place: string;
  country: string;
  coordinates: [number, number];
  visits: string[];
  description: string;
  images?: string[];
  mapCountry?: string;
};

export type TravelTimelineEntry = {
  period: string;
  summary: string;
  places: string[];
};

export type TravelTimelineYear = {
  year: string;
  entries: TravelTimelineEntry[];
};

export const countryOrder = [
  "Singapore",
  "United States",
  "Malaysia",
  "Thailand",
  "United Kingdom",
  "Indonesia",
  "China",
  "Taiwan",
  "Vietnam",
  "France",
  "Monaco",
  "Spain",
  "Qatar",
  "Finland",
  "Japan",
  "Greece",
  "United Arab Emirates",
  "Netherlands",
  "Hong Kong",
  "Iran",
] as const;

export const countryNotes: Record<string, string> = {
  Singapore:
    "Home base, recurring return point, and a little red dot that keeps becoming more personal.",
  "United States":
    "Bay Area workdays and city walks, with a giant-tree detour through Yosemite.",
  Malaysia:
    "Many border crossings and weekend chapters, from Malacca and Johor to Penang, Putrajaya, and Kuala Lumpur.",
  Thailand:
    "Beaches, southern towns, Bangkok energy, and a memorable climbing day above Krabi’s turquoise water.",
  "United Kingdom":
    "A London business trip with blue-hour walks, red telephone boxes, and Big Ben in the background.",
  Indonesia:
    "A night climb to Ijen Crater and a waterfall-filled East Java adventure.",
  China:
    "The longest-running map: hometown chapters, family visits, mountain trips, coastlines, and years of train windows.",
  Taiwan:
    "Two Taipei chapters and a quieter side trip to Yilan.",
  Vietnam:
    "A December visit to the warm, kinetic streets of Ho Chi Minh City.",
  France:
    "Mediterranean light and a stop in Nice along the Côte d’Azur.",
  Monaco:
    "A tiny, polished city-state folded into an October Mediterranean route.",
  Spain:
    "Barcelona for Gaudí lines, late dinners, and a city made for walking.",
  Qatar:
    "An airport-transit chapter connecting longer journeys in Europe and the Gulf.",
  Finland:
    "Helsinki’s sea air, calm geometry, thoughtful design, and Nordic light.",
  Japan:
    "Tokyo’s motion followed by Hakone’s quiet and views toward Mount Fuji.",
  Greece:
    "Athens in winter: ancient stone, bright skies, and mythology at street level.",
  "United Arab Emirates":
    "Dubai and Abu Dhabi—desert light, ambitious architecture, and grand public spaces.",
  Netherlands:
    "A winter Netherlands chapter, remembered through Amsterdam’s canals and narrow streets.",
  "Hong Kong":
    "A compact city chapter of steep streets, dense skylines, and harbor light.",
  Iran:
    "A transit-only footprint—part of the route rather than a full destination chapter.",
};

export const travelLocations: TravelLocation[] = [
  {
    place: "Singapore",
    country: "Singapore",
    coordinates: [103.8198, 1.3521],
    visits: ["Jun 2026", "Dec 2025", "Oct 2022"],
    description:
      "Home, repeat arrivals, tropical rain, skyline walks, and one very enthusiastic Universal Studios day.",
    images: ["/travel/Singapore - Singapore.jpg"],
  },
  {
    place: "Yosemite",
    country: "United States",
    mapCountry: "United States of America",
    coordinates: [-119.5383, 37.8651],
    visits: ["May 2026"],
    description:
      "Sequoias, granite, and the useful reminder that some trees are too big for one hug.",
    images: ["/photos/travel/yosemite.jpeg"],
  },
  {
    place: "San Jose",
    country: "United States",
    mapCountry: "United States of America",
    coordinates: [-121.8863, 37.3382],
    visits: ["May 2026"],
    description:
      "A May South Bay stop, with a flower-framed lunch break tucked between California workdays.",
    images: ["/photos/travel/san-jose.jpeg"],
  },
  {
    place: "Palo Alto",
    country: "United States",
    mapCountry: "United States of America",
    coordinates: [-122.143, 37.4419],
    visits: ["May 2026"],
    description: "A spring Bay Area chapter around Palo Alto.",
  },
  {
    place: "San Francisco",
    country: "United States",
    mapCountry: "United States of America",
    coordinates: [-122.4194, 37.7749],
    visits: ["May 2026", "Apr 2026"],
    description:
      "Golden Gate views, long walks, blue-hour dinners, and more hills than expected.",
    images: [
      "/photos/travel/san-francisco.jpeg",
      "/photos/travel/san-francisco-evening.jpeg",
    ],
  },
  {
    place: "Menlo Park",
    country: "United States",
    mapCountry: "United States of America",
    coordinates: [-122.1817, 37.453],
    visits: ["Apr 2026"],
    description:
      "A California work chapter at Meta’s 1 Hacker Way campus in Menlo Park.",
    images: ["/photos/travel/menlo-park.jpeg"],
  },
  {
    place: "Malacca",
    country: "Malaysia",
    coordinates: [102.2501, 2.1896],
    visits: ["May 2026"],
    description: "Heritage streets, river walks, and a food-first stop.",
  },
  {
    place: "Johor",
    country: "Malaysia",
    coordinates: [103.7414, 1.4927],
    visits: ["During 2025"],
    description: "A familiar cross-border chapter in southern Malaysia.",
    images: ["/travel/Johor Bahru - Malaysia.jpg"],
  },
  {
    place: "Kuala Lumpur",
    country: "Malaysia",
    coordinates: [101.6869, 3.139],
    visits: ["Mar 2024"],
    description: "A city break of food, towers, and humid evening walks.",
    images: ["/travel/Kuala Lumpur - Malaysia.jpg"],
  },
  {
    place: "Padang Besar",
    country: "Malaysia",
    coordinates: [100.3217, 6.661],
    visits: ["Apr–May 2024"],
    description: "A northern border stop on the Thailand–Malaysia route.",
  },
  {
    place: "Batu Anam",
    country: "Malaysia",
    coordinates: [102.703, 2.512],
    visits: ["Date not recorded"],
    description: "An undated footprint in Johor.",
  },
  {
    place: "Bayan Lepas",
    country: "Malaysia",
    coordinates: [100.264, 5.297],
    visits: ["Date not recorded"],
    description: "A Penang stop near the island’s southern edge.",
  },
  {
    place: "Kangar",
    country: "Malaysia",
    coordinates: [100.1986, 6.4414],
    visits: ["Date not recorded"],
    description: "An undated northern Malaysia stop.",
  },
  {
    place: "Tanjung Tokong",
    country: "Malaysia",
    coordinates: [100.305, 5.46],
    visits: ["Date not recorded"],
    description: "A coastal Penang footprint.",
  },
  {
    place: "Butterworth",
    country: "Malaysia",
    coordinates: [100.363, 5.399],
    visits: ["Date not recorded"],
    description: "A mainland Penang stop across the channel.",
  },
  {
    place: "Batu Ferringhi",
    country: "Malaysia",
    coordinates: [100.245, 5.475],
    visits: ["Date not recorded"],
    description: "Beach roads and a slower Penang coastline.",
  },
  {
    place: "George Town",
    country: "Malaysia",
    coordinates: [100.3327, 5.4141],
    visits: ["Date not recorded"],
    description: "Heritage shophouses, street art, and excellent food.",
  },
  {
    place: "Gelugor",
    country: "Malaysia",
    coordinates: [100.303, 5.355],
    visits: ["Date not recorded"],
    description: "Another Penang chapter, tucked between city and coast.",
  },
  {
    place: "Nilai",
    country: "Malaysia",
    coordinates: [101.797, 2.819],
    visits: ["Date not recorded"],
    description: "An undated stop south of Kuala Lumpur.",
  },
  {
    place: "Putrajaya",
    country: "Malaysia",
    coordinates: [101.6964, 2.9264],
    visits: ["Date not recorded"],
    description: "Lakes, bridges, and Malaysia’s planned administrative city.",
  },
  {
    place: "Sepang",
    country: "Malaysia",
    coordinates: [101.749, 2.691],
    visits: ["Date not recorded"],
    description: "A travel-hub footprint at the edge of the Klang Valley.",
  },
  {
    place: "Krabi",
    country: "Thailand",
    coordinates: [98.9063, 8.0863],
    visits: ["May 2026", "Apr–May 2024"],
    description:
      "Beach days and a climbing route with an impossible turquoise view.",
    images: ["/photos/travel/krabi-climbing.jpeg"],
  },
  {
    place: "Trang",
    country: "Thailand",
    coordinates: [99.6114, 7.5563],
    visits: ["Apr–May 2024"],
    description: "A quieter southern Thailand stop beyond the usual route.",
  },
  {
    place: "Bangkok",
    country: "Thailand",
    coordinates: [100.5018, 13.7563],
    visits: ["Date not recorded"],
    description: "Markets, temples, traffic, and streets that stay awake.",
    images: ["/travel/Bangkok - Thailand.jpg"],
  },
  {
    place: "Hat Yai",
    country: "Thailand",
    coordinates: [100.4747, 7.0084],
    visits: ["Date not recorded"],
    description: "A southern city stop near the Malaysian border.",
  },
  {
    place: "Ao Nang",
    country: "Thailand",
    coordinates: [98.8307, 8.034],
    visits: ["Date not recorded"],
    description: "Limestone cliffs, long-tail boats, and Andaman sunsets.",
  },
  {
    place: "London",
    country: "United Kingdom",
    coordinates: [-0.1276, 51.5072],
    visits: ["Mar 2026"],
    description: "A business trip with time saved for a blue-hour city walk.",
    images: ["/photos/travel/london.jpeg"],
  },
  {
    place: "East Java · Ijen & waterfalls",
    country: "Indonesia",
    coordinates: [114.242, -8.058],
    visits: ["Dec 2025"],
    description:
      "A sulfur-mask night at Ijen Crater followed by East Java’s towering waterfalls.",
    images: [
      "/photos/travel/ijen-crater.jpeg",
      "/photos/travel/east-java-waterfall.jpeg",
    ],
  },
  {
    place: "Jinan",
    country: "China",
    coordinates: [117.1201, 36.6512],
    visits: ["Nov 2025"],
    description: "A late-autumn return to Shandong’s City of Springs.",
    images: ["/travel/Jinan - China.jpg"],
  },
  {
    place: "Dali",
    country: "China",
    coordinates: [100.2676, 25.6065],
    visits: ["Sep 2025"],
    description: "Cangshan, Erhai, and clear Yunnan light.",
  },
  {
    place: "Guiyang",
    country: "China",
    coordinates: [106.6302, 26.6477],
    visits: ["Jan 2025"],
    description: "A winter visit in Guizhou’s mountain capital.",
  },
  {
    place: "Suqian",
    country: "China",
    coordinates: [118.2752, 33.963],
    visits: ["Jan 2025"],
    description: "A January family-and-city chapter in Jiangsu.",
  },
  {
    place: "Xiamen",
    country: "China",
    coordinates: [118.0894, 24.4798],
    visits: ["Jun 2024"],
    description: "Sea air, island walks, and southern Fujian streets.",
  },
  {
    place: "Shenzhen",
    country: "China",
    coordinates: [114.0579, 22.5431],
    visits: ["Dec 2023"],
    description: "A fast-moving December stop in Guangdong.",
  },
  {
    place: "Guangzhou",
    country: "China",
    coordinates: [113.2644, 23.1291],
    visits: ["Aug 2022"],
    description: "A summer visit shaped by food, family, and Pearl River heat.",
  },
  {
    place: "Foshan",
    country: "China",
    coordinates: [113.1214, 23.0218],
    visits: ["Aug 2022"],
    description: "A neighboring Guangdong chapter alongside Guangzhou.",
  },
  {
    place: "Tianjin",
    country: "China",
    coordinates: [117.2, 39.0842],
    visits: ["2021", "2019", "2018"],
    description: "A city returned to across several years.",
    images: ["/travel/Tianjin - China.jpg"],
  },
  {
    place: "Hangzhou",
    country: "China",
    coordinates: [120.1551, 30.2741],
    visits: ["Oct 2020"],
    description: "West Lake in autumn and a softer city rhythm.",
    images: ["/travel/Hangzhou - China.jpg"],
  },
  {
    place: "Chengdu",
    country: "China",
    coordinates: [104.0665, 30.5728],
    visits: ["Jan 2020"],
    description: "Winter in Sichuan, spicy meals, and the start of a western route.",
    images: ["/travel/Chengdu - China.jpg"],
  },
  {
    place: "Aba",
    country: "China",
    coordinates: [102.2247, 31.8994],
    visits: ["Jan 2020"],
    description: "High-altitude landscapes in the Aba Tibetan and Qiang region.",
  },
  {
    place: "Tai’an",
    country: "China",
    coordinates: [117.0876, 36.2003],
    visits: ["Aug 2019", "Jul 2016"],
    description: "Two visits and the enduring vertical challenge of Mount Tai.",
    images: ["/travel/Tai'an - China.jpg"],
  },
  {
    place: "Zhangjiajie",
    country: "China",
    coordinates: [110.4792, 29.1171],
    visits: ["Date not recorded"],
    description: "Sandstone pillars, mountain paths, and misty views.",
  },
  {
    place: "Qingdao",
    country: "China",
    coordinates: [120.3826, 36.0671],
    visits: ["Date not recorded"],
    description: "A coastal Shandong footprint.",
  },
  {
    place: "Xuzhou",
    country: "China",
    coordinates: [117.2841, 34.2058],
    visits: ["Date not recorded"],
    description: "An undated Jiangsu city chapter.",
    images: ["/travel/Xuzhou - China.jpg"],
  },
  {
    place: "Nanjing",
    country: "China",
    coordinates: [118.7969, 32.0603],
    visits: ["Date not recorded"],
    description: "Plane trees, history, and long walks through an old capital.",
    images: ["/travel/Nanjing - China.jpg"],
  },
  {
    place: "Shanghai",
    country: "China",
    coordinates: [121.4737, 31.2304],
    visits: ["Date not recorded"],
    description: "A recurring reference point of skyline, lanes, and city energy.",
    images: ["/travel/Shanghai - China.jpg"],
  },
  {
    place: "Taipei",
    country: "Taiwan",
    coordinates: [121.5654, 25.033],
    visits: ["Aug 2025", "Dec 2023"],
    description: "Two visits for night markets, city wandering, and mountain edges.",
  },
  {
    place: "Yilan",
    country: "Taiwan",
    coordinates: [121.753, 24.757],
    visits: ["Dec 2023"],
    description: "A quieter northeastern Taiwan side trip.",
  },
  {
    place: "Ho Chi Minh City",
    country: "Vietnam",
    coordinates: [106.6297, 10.8231],
    visits: ["Dec 2024"],
    description: "Warm December nights, scooters, cafés, and constant motion.",
  },
  {
    place: "Nice",
    country: "France",
    coordinates: [7.262, 43.7102],
    visits: ["Oct 2023"],
    description: "Mediterranean blue along the Promenade des Anglais.",
  },
  {
    place: "Monaco",
    country: "Monaco",
    coordinates: [7.4246, 43.7384],
    visits: ["Oct 2023"],
    description: "A tiny city-state stop between Riviera chapters.",
  },
  {
    place: "Barcelona",
    country: "Spain",
    coordinates: [2.1734, 41.3851],
    visits: ["Oct 2023"],
    description: "Gaudí curves, late dinners, and a walkable Mediterranean grid.",
  },
  {
    place: "Airport transit",
    country: "Qatar",
    coordinates: [51.6138, 25.2731],
    visits: ["Oct 2023", "Jan 2023"],
    description: "Two airport connections; a transit footprint rather than a full trip.",
  },
  {
    place: "Helsinki",
    country: "Finland",
    coordinates: [24.9384, 60.1699],
    visits: ["May 2023"],
    description: "Sea air, Nordic design, and a calm spring city rhythm.",
    images: ["/travel/Helsinki - Finland.jpg"],
  },
  {
    place: "Tokyo",
    country: "Japan",
    coordinates: [139.6917, 35.6895],
    visits: ["Apr 2023"],
    description: "A kinetic first chapter of neighborhoods, trains, and tiny details.",
    images: ["/travel/Tokyo - Japan.jpg"],
  },
  {
    place: "Hakone · Mount Fuji",
    country: "Japan",
    coordinates: [139.1069, 35.2324],
    visits: ["Apr 2023"],
    description: "Hot springs, mountain air, and a Fuji-view detour from Tokyo.",
    images: ["/travel/Hakone - Japan.jpg"],
  },
  {
    place: "Athens",
    country: "Greece",
    coordinates: [23.7275, 37.9838],
    visits: ["Jan 2023"],
    description: "Ancient stone and bright winter skies in the Greek capital.",
    images: ["/travel/Athens - Greece - 1.jpg"],
  },
  {
    place: "Dubai",
    country: "United Arab Emirates",
    coordinates: [55.2708, 25.2048],
    visits: ["Jan 2023"],
    description: "Desert light, waterfront walks, and a skyline turned all the way up.",
    images: ["/travel/Dubai - United Arab Emirates.jpg"],
  },
  {
    place: "Abu Dhabi",
    country: "United Arab Emirates",
    coordinates: [54.3773, 24.4539],
    visits: ["Jan 2023"],
    description: "Grand architecture and a calmer Gulf-city pace.",
    images: ["/travel/Abu Dhabi - United Arab Emirates.jpg"],
  },
  {
    place: "Amsterdam",
    country: "Netherlands",
    coordinates: [4.9041, 52.3676],
    visits: ["Jan 2023"],
    description: "Canals, narrow houses, bicycles, and a winter city best read on foot.",
    images: ["/travel/Amsterdam - Netherlands.jpg"],
  },
  {
    place: "Hong Kong",
    country: "Hong Kong",
    coordinates: [114.1694, 22.3193],
    visits: ["Date not recorded"],
    description: "An undated city chapter of harbor views and steep streets.",
  },
  {
    place: "Transit stop",
    country: "Iran",
    coordinates: [51.1347, 35.4161],
    visits: ["Date not recorded"],
    description: "Recorded as passing through, not as a destination stay.",
  },
];

export const travelTimeline: TravelTimelineYear[] = [
  {
    year: "2026",
    entries: [
      {
        period: "June",
        summary: "Singapore",
        places: ["Singapore"],
      },
      {
        period: "May",
        summary:
          "United States, Malaysia, and Thailand",
        places: [
          "Yosemite",
          "San Jose",
          "Palo Alto",
          "San Francisco",
          "Malacca",
          "Krabi · climbing",
        ],
      },
      {
        period: "April",
        summary: "United States",
        places: ["San Francisco", "Menlo Park, California"],
      },
      {
        period: "March",
        summary: "United Kingdom · business trip",
        places: ["London"],
      },
    ],
  },
  {
    year: "2025",
    entries: [
      {
        period: "December",
        summary: "Indonesia and Singapore",
        places: ["East Java · Ijen Crater", "Singapore"],
      },
      {
        period: "November",
        summary: "China",
        places: ["Jinan, Shandong"],
      },
      {
        period: "September",
        summary: "China",
        places: ["Dali, Yunnan"],
      },
      {
        period: "August",
        summary: "Taiwan",
        places: ["Taipei"],
      },
      {
        period: "January",
        summary: "China",
        places: ["Guiyang, Guizhou", "Suqian, Jiangsu"],
      },
      {
        period: "During the year",
        summary: "Malaysia",
        places: ["Johor and nearby stops"],
      },
    ],
  },
  {
    year: "2024",
    entries: [
      {
        period: "December",
        summary: "Vietnam",
        places: ["Ho Chi Minh City"],
      },
      {
        period: "June",
        summary: "China",
        places: ["Xiamen, Fujian"],
      },
      {
        period: "April–May",
        summary: "Thailand and Malaysia",
        places: ["Krabi beaches", "Trang", "Padang Besar"],
      },
      {
        period: "March",
        summary: "Malaysia",
        places: ["Kuala Lumpur"],
      },
    ],
  },
  {
    year: "2023",
    entries: [
      {
        period: "December",
        summary: "Taiwan and China",
        places: ["Taipei", "Yilan", "Shenzhen, Guangdong"],
      },
      {
        period: "October",
        summary: "France, Monaco, Spain, and Qatar transit",
        places: ["Nice", "Monaco", "Barcelona", "Qatar airport"],
      },
      {
        period: "May",
        summary: "Finland",
        places: ["Helsinki"],
      },
      {
        period: "April",
        summary: "Japan",
        places: ["Tokyo", "Hakone · Mount Fuji"],
      },
      {
        period: "January",
        summary: "Greece, Qatar transit, UAE, and the Netherlands",
        places: [
          "Athens",
          "Qatar airport",
          "Dubai",
          "Abu Dhabi",
          "Amsterdam",
        ],
      },
    ],
  },
  {
    year: "2022",
    entries: [
      {
        period: "October",
        summary: "Singapore",
        places: ["Universal Studios Singapore"],
      },
      {
        period: "August",
        summary: "China",
        places: ["Guangzhou", "Foshan"],
      },
    ],
  },
  {
    year: "2021 & earlier",
    entries: [
      {
        period: "2021 · 2019 · 2018",
        summary: "China",
        places: ["Tianjin"],
      },
      {
        period: "October 2020",
        summary: "China",
        places: ["Hangzhou, Zhejiang"],
      },
      {
        period: "January 2020",
        summary: "China",
        places: ["Chengdu, Sichuan", "Aba, Sichuan"],
      },
      {
        period: "August 2019 · July 2016",
        summary: "China",
        places: ["Tai’an, Shandong"],
      },
    ],
  },
];

export const undatedTravelGroups = [
  {
    country: "China",
    places: [
      "Zhangjiajie",
      "Qingdao",
      "Xuzhou",
      "Nanjing",
      "Shanghai",
    ],
  },
  {
    country: "Malaysia",
    places: [
      "Batu Anam",
      "Bayan Lepas",
      "Kangar",
      "Tanjung Tokong",
      "Butterworth",
      "Batu Ferringhi",
      "George Town",
      "Gelugor",
      "Nilai",
      "Putrajaya",
      "Sepang",
    ],
  },
  {
    country: "Thailand",
    places: ["Bangkok", "Hat Yai", "Ao Nang"],
  },
  {
    country: "Other countries & regions",
    places: ["Hong Kong", "Iran · transit only"],
  },
];

export const featuredTravelMoments = [
  {
    image: "/photos/travel/menlo-park.jpeg",
    place: "Menlo Park · Meta",
    date: "Apr 2026",
    alt: "Huiyu standing in front of the Meta sign at 1 Hacker Way in Menlo Park",
  },
  {
    image: "/photos/travel/san-jose.jpeg",
    place: "San Jose",
    date: "May 2026",
    alt: "A flower-framed restaurant entrance in San Jose",
  },
  {
    image: "/photos/travel/yosemite.jpeg",
    place: "Yosemite",
    date: "May 2026",
    alt: "Huiyu hugging a giant sequoia in Yosemite",
  },
  {
    image: "/photos/travel/krabi-climbing.jpeg",
    place: "Krabi · climbing",
    date: "May 2026",
    alt: "Huiyu rock climbing above a turquoise bay in Krabi",
  },
  {
    image: "/photos/travel/london.jpeg",
    place: "London",
    date: "Mar 2026",
    alt: "Huiyu beside a red telephone box with Big Ben behind her",
  },
  {
    image: "/photos/travel/ijen-crater.jpeg",
    place: "Ijen Crater",
    date: "Dec 2025",
    alt: "Huiyu in a sulfur mask at Ijen Crater at night",
  },
  {
    image: "/photos/travel/east-java-waterfall.jpeg",
    place: "East Java",
    date: "Dec 2025",
    alt: "Huiyu standing beneath a tall waterfall in East Java",
  },
  {
    image: "/photos/travel/san-francisco.jpeg",
    place: "San Francisco",
    date: "May 2026",
    alt: "Huiyu in front of the Golden Gate Bridge",
  },
  {
    image: "/photos/travel/san-francisco-evening.jpeg",
    place: "San Francisco · blue hour",
    date: "May 2026",
    alt: "Huiyu having a drink by the San Francisco waterfront at blue hour",
  },
];
