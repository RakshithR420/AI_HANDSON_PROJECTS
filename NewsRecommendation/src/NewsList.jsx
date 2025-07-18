import NewsItem from "./NewsItem";
export default function NewsList({ recommended_list }) {
  return (
    <div>
      <ul className="flex flex-col gap-4 border border-gray-300 rounded-md p-4">
        {recommended_list.map((el) => (
          <li key={el.NewsID} className="">
            <NewsItem Item={el} />
          </li>
        ))}
      </ul>
    </div>
  );
}

// flex flex-col gap-4 px-4
