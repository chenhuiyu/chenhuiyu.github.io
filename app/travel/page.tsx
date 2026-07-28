import type { Metadata } from "next";
import { SiteFooter, SiteHeader } from "@/app/components/SiteHeader";
import { countryOrder, travelLocations } from "@/content/travel-data";
import { TravelGlobe } from "./TravelGlobe";

export const metadata: Metadata = {
  title: "Travel Globe — Huiyu Chen",
  description:
    "An interactive 3D globe and complete travel timeline of the countries, regions, cities, and places Huiyu Chen has visited.",
};

export default function TravelPage() {
  return (
    <main className="site-shell travel-shell" id="top">
      <SiteHeader />
      <section className="travel-hero">
        <div>
          <p className="eyebrow">
            <span>Footprints · 足迹</span>
            <span aria-hidden="true">·</span>
            tiny planet, many stairs
          </p>
          <h1>
            Places that made my
            <br />
            world a little <span className="word-mark">bigger</span>.
          </h1>
        </div>
        <div className="travel-hero-note">
          <span aria-hidden="true">✈︎</span>
          <p>
            {countryOrder.length} countries &amp; regions ·{" "}
            {travelLocations.length} places
          </p>
          <small>
            Spin the globe, tap a glowing place, then follow the complete
            2016–2026 timeline.
          </small>
        </div>
      </section>
      <TravelGlobe />
      <SiteFooter />
    </main>
  );
}
