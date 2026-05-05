module.exports = {
  branches: ["main"],
  tagFormat: "v${version}",
  plugins: [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    [
      "@semantic-release/exec",
      {
        prepareCmd: "scripts/build_macos.sh ${nextRelease.version}"
      }
    ],
    [
      "@semantic-release/github",
      {
        assets: [
          {
            path: "dist/release/*.zip",
            label: "EkoVideo Compressor macOS Apple Silicon"
          }
        ],
        successComment: false,
        failComment: false
      }
    ]
  ]
};
