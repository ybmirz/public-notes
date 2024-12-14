import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "📖 miru's notes",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "notes.mirzahiday.at",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    generateSocialImages: false,
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#cdd6f4", // Text
          lightgray: "#bac2de", // Subtext0
          gray: "#a6adc8", // Subtext1
          darkgray: "#585b70", // Surface2
          dark: "#313244", // Surface0
          secondary: "#89b4fa", // Blue
          tertiary: "#94e2d5", // Teal
          highlight: "rgba(137, 180, 250, 0.15)", // Blue with opacity
          textHighlight: "#f9e2af88", // Yellow with opacity
        },
        darkMode: {
          light: "#1e1e2e", // Base
          lightgray: "#313244", // Surface0
          gray: "#45475a", // Surface1
          darkgray: "#cdd6f4", // Text
          dark: "#f5e0dc", // Rosewater
          secondary: "#89b4fa", // Blue
          tertiary: "#94e2d5", // Teal
          highlight: "rgba(137, 180, 250, 0.15)", // Blue with opacity
          textHighlight: "#f9e2af88", // Yellow with opacity
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
    ],
  },
}

export default config
