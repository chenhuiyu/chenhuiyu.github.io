"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  geoDistance,
  geoGraticule10,
  geoOrthographic,
  geoPath,
} from "d3-geo";
import { feature } from "topojson-client";
import world from "world-atlas/countries-110m.json";
import {
  countryNotes,
  countryOrder,
  featuredTravelMoments,
  travelLocations,
  travelTimeline,
  undatedTravelGroups,
  type TravelLocation,
} from "@/content/travel-data";

type CountryFeature = {
  type: "Feature";
  id?: string | number;
  properties: { name: string };
  geometry: unknown;
};

const countryFeatures = (
  feature(
    world as never,
    (world as { objects: { countries: unknown } }).objects.countries as never,
  ) as unknown as { features: CountryFeature[] }
).features;

function mapCountryOf(location: TravelLocation) {
  return location.mapCountry ?? location.country;
}

export function TravelGlobe() {
  const [rotation, setRotation] = useState<[number, number]>([-104, -18]);
  const [selected, setSelected] = useState("Singapore");
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{
    x: number;
    y: number;
    rotation: [number, number];
  } | null>(null);

  const countries = useMemo(() => {
    const grouped = new Map<string, TravelLocation[]>();
    for (const country of countryOrder) grouped.set(country, []);
    for (const location of travelLocations) {
      grouped.set(location.country, [
        ...(grouped.get(location.country) ?? []),
        location,
      ]);
    }
    return grouped;
  }, []);

  const displayCountryForMap = useMemo(() => {
    const mapping = new Map<string, string>();
    for (const location of travelLocations) {
      const mapCountry = mapCountryOf(location);
      if (!mapping.has(mapCountry)) mapping.set(mapCountry, location.country);
    }
    return mapping;
  }, []);

  const visitedMapNames = useMemo(
    () => new Set(displayCountryForMap.keys()),
    [displayCountryForMap],
  );

  useEffect(() => {
    if (dragging) return;
    const timer = window.setInterval(() => {
      setRotation(([longitude, latitude]) => [longitude + 0.28, latitude]);
    }, 40);
    return () => window.clearInterval(timer);
  }, [dragging]);

  const projection = useMemo(
    () =>
      geoOrthographic()
        .fitExtent(
          [
            [20, 20],
            [640, 640],
          ],
          { type: "Sphere" },
        )
        .rotate(rotation),
    [rotation],
  );
  const path = geoPath(projection);
  const selectedLocations = countries.get(selected) ?? [];
  const selectedTripPeriods = new Set(
    selectedLocations.flatMap((location) => location.visits),
  ).size;

  return (
    <div className="travel-experience">
      <div className="globe-card">
        <div className="globe-copy">
          <p className="globe-label">Drag me around · 转转看</p>
          <h2>{selected}</h2>
          <p>{countryNotes[selected]}</p>
          <div className="country-mini-stats">
            <span>
              <strong>{selectedLocations.length}</strong> places
            </span>
            <span>
              <strong>{selectedTripPeriods}</strong> trip periods
            </span>
          </div>
        </div>

        <svg
          aria-label="Interactive globe showing countries and regions Huiyu has visited"
          className="travel-globe"
          onPointerDown={(event) => {
            setDragging(true);
            dragStart.current = {
              x: event.clientX,
              y: event.clientY,
              rotation,
            };
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            if (!dragStart.current) return;
            const deltaX = event.clientX - dragStart.current.x;
            const deltaY = event.clientY - dragStart.current.y;
            setRotation([
              dragStart.current.rotation[0] + deltaX * 0.35,
              Math.max(
                -55,
                Math.min(
                  55,
                  dragStart.current.rotation[1] - deltaY * 0.25,
                ),
              ),
            ]);
          }}
          onPointerUp={(event) => {
            dragStart.current = null;
            setDragging(false);
            event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          onPointerCancel={() => {
            dragStart.current = null;
            setDragging(false);
          }}
          role="img"
          viewBox="0 0 660 660"
        >
          <defs>
            <radialGradient id="ocean" cx="35%" cy="28%">
              <stop offset="0%" stopColor="#dcebe8" />
              <stop offset="72%" stopColor="#aacbc8" />
              <stop offset="100%" stopColor="#6f9d9a" />
            </radialGradient>
            <filter
              id="countryGlow"
              x="-30%"
              y="-30%"
              width="160%"
              height="160%"
            >
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <path className="globe-ocean" d={path({ type: "Sphere" }) ?? ""} />
          <path className="globe-graticule" d={path(geoGraticule10()) ?? ""} />
          {countryFeatures.map((country) => {
            const mapName = country.properties.name;
            const displayCountry = displayCountryForMap.get(mapName);
            const isVisited = visitedMapNames.has(mapName);
            const isSelected = selected === displayCountry;
            return (
              <path
                aria-label={
                  isVisited
                    ? `${displayCountry}, visited`
                    : `${mapName}, not visited`
                }
                className={[
                  "globe-country",
                  isVisited ? "is-visited" : "",
                  isSelected ? "is-selected" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                d={path(country as never) ?? ""}
                key={country.id ?? mapName}
                onClick={(event) => {
                  event.stopPropagation();
                  if (displayCountry) setSelected(displayCountry);
                }}
                onKeyDown={(event) => {
                  if (
                    displayCountry &&
                    (event.key === "Enter" || event.key === " ")
                  ) {
                    event.preventDefault();
                    setSelected(displayCountry);
                  }
                }}
                role={isVisited ? "button" : undefined}
                tabIndex={isVisited ? 0 : undefined}
              />
            );
          })}
          {travelLocations.map((location) => {
            const visible =
              geoDistance(
                location.coordinates,
                [-rotation[0], -rotation[1]],
              ) <
              Math.PI / 2;
            const point = projection(location.coordinates);
            if (!visible || !point) return null;
            return (
              <g
                aria-label={`${location.place}, ${location.country}`}
                className={
                  selected === location.country
                    ? "city-point is-selected"
                    : "city-point"
                }
                key={`${location.place}-${location.country}`}
                onClick={(event) => {
                  event.stopPropagation();
                  setSelected(location.country);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    setSelected(location.country);
                  }
                }}
                role="button"
                tabIndex={0}
                transform={`translate(${point[0]} ${point[1]})`}
              >
                <circle r="8" />
                <circle r="2.8" />
              </g>
            );
          })}
        </svg>
      </div>

      <div className="country-pills" aria-label="Visited countries and regions">
        {countryOrder.map((country) => (
          <button
            className={selected === country ? "active" : ""}
            key={country}
            onClick={() => setSelected(country)}
            type="button"
          >
            <span aria-hidden="true">✦</span> {country}
          </button>
        ))}
      </div>

      <section className="travel-photo-diary" aria-labelledby="photo-diary-title">
        <div className="travel-photo-heading">
          <p className="section-kicker">Recent frames · 最近几格</p>
          <h2 id="photo-diary-title">
            A few moments that deserved more than a map pin.
          </h2>
          <p>
            Giant trees, high cliffs, sulfur masks, city lights—and proof that
            I occasionally stop moving long enough for a photo.
          </p>
        </div>
        <div className="travel-photo-grid">
          {featuredTravelMoments.map((moment, index) => (
            <figure
              className={`travel-photo travel-photo-${index + 1}`}
              key={moment.image}
            >
              <img alt={moment.alt} src={moment.image} />
              <figcaption>
                <span>{moment.place}</span>
                <small>{moment.date}</small>
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="travel-detail" aria-live="polite">
        <div className="travel-detail-heading">
          <p className="section-kicker">Postcards from · 来自</p>
          <h2>{selected}</h2>
          <p>{selectedLocations.map((location) => location.place).join(" · ")}</p>
        </div>
        <div className="postcard-grid">
          {selectedLocations.map((location, index) => (
            <article
              className={[
                "postcard",
                `postcard-${(index % 3) + 1}`,
                location.images?.length ? "" : "no-image",
              ]
                .filter(Boolean)
                .join(" ")}
              key={`${location.place}-${location.country}`}
            >
              {location.images?.length ? (
                <div
                  className={
                    location.images.length > 1
                      ? "postcard-photo has-pair"
                      : "postcard-photo"
                  }
                >
                  {location.images.slice(0, 2).map((image, imageIndex) => (
                    <img
                      alt={
                        imageIndex === 0
                          ? `${location.place}, ${selected}`
                          : `Another moment in ${location.place}, ${selected}`
                      }
                      key={image}
                      src={image}
                    />
                  ))}
                </div>
              ) : null}
              <div className="postcard-copy">
                <p>{String(index + 1).padStart(2, "0")}</p>
                <h3>{location.place}</h3>
                <p>{location.description}</p>
                <span>{location.visits.join(" · ")}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="travel-timeline" aria-labelledby="travel-timeline-title">
        <div className="travel-timeline-heading">
          <p className="section-kicker">Travel log · 完整时间轴</p>
          <h2 id="travel-timeline-title">
            When each pin found its way onto the globe.
          </h2>
          <p>
            The dated record from 2016 to 2026, followed by places whose exact
            dates are still hiding somewhere in the camera roll.
          </p>
        </div>

        <div className="travel-years">
          {travelTimeline.map((group) => (
            <article className="travel-year" key={group.year}>
              <h3>{group.year}</h3>
              <div>
                {group.entries.map((entry) => (
                  <div
                    className="travel-timeline-entry"
                    key={`${group.year}-${entry.period}`}
                  >
                    <p>{entry.period}</p>
                    <div>
                      <h4>{entry.summary}</h4>
                      <p>{entry.places.join(" · ")}</p>
                    </div>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </div>

        <div className="undated-travel">
          <div>
            <p className="section-kicker">Date TBD · 时间待考古</p>
            <h3>Also on the map</h3>
          </div>
          <div className="undated-grid">
            {undatedTravelGroups.map((group) => (
              <article key={group.country}>
                <h4>{group.country}</h4>
                <p>{group.places.join(" · ")}</p>
              </article>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
