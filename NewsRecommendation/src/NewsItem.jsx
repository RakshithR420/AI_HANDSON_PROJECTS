import React from "react";

export default function NewsItem({ Item }) {
  return (
    <div className="border border-gray-600 rounded-md text-left p-4 space-y-3 space-x-3 hover:bg-gray-950 hover:shadow-md transition-all">
      <h2 className="font-bold text-3xl">{Item.Title}</h2>
      <p className="bg-cyan-700 rounded-full inline-block p-1 text-base font-medium ">
        {Item.Category}
      </p>
      <p className="bg-lime-700 rounded-full inline-block p-1 text-base font-medium">
        {Item.SubCategory}
      </p>
      <p>{Item.Abstract}</p>
      <a href={Item.URL} className="text-blue-500 underline mt-2">
        Read More
      </a>
      <p className="text-right text-gray-600">{Item.NewsID}</p>
    </div>
  );
}
