// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "EkoVideoCompressor",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "EkoVideoCompressor", targets: ["EkoVideoCompressor"])
    ],
    targets: [
        .executableTarget(
            name: "EkoVideoCompressor",
            path: "EkoVideoCompressor"
        )
    ]
)
